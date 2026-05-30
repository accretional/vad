package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"google.golang.org/protobuf/encoding/prototext"

	"github.com/accretional/vad/internal/embedded"
	"github.com/accretional/vad/internal/server"
	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
)

// onDiskWeightsRoot is the conventional directory the server checks for
// per-backend on-disk weights (used as a fallback when something isn't
// embedded; e.g. a model added after this binary was built).
const onDiskWeightsRoot = "weights"

// backendDirName maps a VADModel enum to the directory name under weights/
// that holds its model.onnx (and any auxiliary files). Used by both the
// startup loader and the Fetch RPC's per-model lookup.
func backendDirName(m pb.VADModel) (string, bool) {
	switch m {
	case pb.VADModel_VAD_MODEL_PYANNOTE:
		return "pyannote", true
	case pb.VADModel_VAD_MODEL_FSMN:
		return "fsmn-vad", true
	case pb.VADModel_VAD_MODEL_FIRERED:
		return "firered-vad", true
	case pb.VADModel_VAD_MODEL_MARBLENET:
		return "marblenet", true
	case pb.VADModel_VAD_MODEL_SILERO:
		return "silero", true
	}
	return "", false
}

// legacyBackendName converts the deprecated -backend string into a VADModel.
func legacyBackendName(s string) pb.VADModel {
	switch s {
	case "":
		return pb.VADModel_VAD_MODEL_UNSPECIFIED
	case "pyannote":
		return pb.VADModel_VAD_MODEL_PYANNOTE
	case "fsmn":
		return pb.VADModel_VAD_MODEL_FSMN
	case "firered":
		return pb.VADModel_VAD_MODEL_FIRERED
	case "marblenet":
		return pb.VADModel_VAD_MODEL_MARBLENET
	case "silero":
		return pb.VADModel_VAD_MODEL_SILERO
	}
	return pb.VADModel_VAD_MODEL_UNSPECIFIED
}

// discoverOnnxRuntime looks for libonnxruntime.{dylib,so} in conventional
// locations and returns the first match, or "" if none found.
//
// **Diagnostic only.** The server always loads the embedded dylib (see
// internal/embedded.MaterializeDylib); this function is used at startup to
// log whether a system install also exists, and to warn if the local copy
// looks like it could conflict (different size suggests different version).
func discoverOnnxRuntime() string {
	var libName string
	switch runtime.GOOS {
	case "darwin":
		libName = "libonnxruntime.dylib"
	case "linux":
		libName = "libonnxruntime.so"
	default:
		return ""
	}
	candidates := []string{}
	if exe, err := os.Executable(); err == nil {
		for _, base := range []string{filepath.Dir(exe), filepath.Dir(filepath.Dir(exe))} {
			matches, _ := filepath.Glob(filepath.Join(base, "third_party", "onnxruntime-*", "lib", libName))
			candidates = append(candidates, matches...)
		}
	}
	if matches, _ := filepath.Glob(filepath.Join("third_party", "onnxruntime-*", "lib", libName)); len(matches) > 0 {
		candidates = append(candidates, matches...)
	}
	candidates = append(candidates,
		"/usr/local/lib/"+libName,
		"/opt/homebrew/lib/"+libName,
		"/usr/lib/"+libName,
		"/usr/lib/x86_64-linux-gnu/"+libName,
		"/usr/lib/aarch64-linux-gnu/"+libName,
	)
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

func main() {
	configPath := flag.String("config", "",
		"path to a VADConfig textproto file (see proto/vad.proto for the schema)")
	backendStr := flag.String("backend", "",
		"DEPRECATED: legacy backend selector (pyannote | fsmn | firered | marblenet | silero). Prefer -config.")
	port := flag.Int("port", 0, "gRPC port (overrides config; default 50051)")
	modelPath := flag.String("model", "",
		"weights dir for the inference backend (overrides config; defaults to weights/<backend>/)")
	libPath := flag.String("lib", "",
		"override the embedded ONNX Runtime with an external libonnxruntime.{dylib,so}. "+
			"Rare — the binary ships with the right dylib for its build target. "+
			"Falls back to ONNXRUNTIME_LIB env var.")
	weightsURL := flag.String("weights-url", "",
		"URL the Fetch RPC redirects clients to (when they request the configured default model)")
	flag.Parse()

	cfg := &pb.VADConfig{}
	if *configPath != "" {
		data, err := os.ReadFile(*configPath)
		if err != nil {
			log.Fatalf("read config %s: %v", *configPath, err)
		}
		if err := prototext.Unmarshal(data, cfg); err != nil {
			log.Fatalf("parse config %s: %v", *configPath, err)
		}
	}
	if v := legacyBackendName(*backendStr); v != pb.VADModel_VAD_MODEL_UNSPECIFIED {
		cfg.Model = v
	}
	if *port != 0 {
		cfg.Port = int32(*port)
	}
	if *modelPath != "" {
		cfg.WeightsDir = *modelPath
	}
	if *libPath != "" {
		cfg.OnnxruntimeLib = *libPath
	}
	if *weightsURL != "" {
		cfg.WeightsUrl = *weightsURL
	}
	if cfg.Model == pb.VADModel_VAD_MODEL_UNSPECIFIED {
		cfg.Model = pb.VADModel_VAD_MODEL_PYANNOTE
	}
	if cfg.Port == 0 {
		cfg.Port = 50051
	}

	// Resolve the ONNX Runtime dylib path. Priority:
	//   1. explicit override (CLI -lib, config field, or env) — for advanced
	//      users pinning a specific version
	//   2. embedded dylib materialized to a temp file (default for prod)
	//   3. discovered system install (back-compat only, deprecated)
	var ortPath, ortTempPath string
	ortExplicit := cfg.OnnxruntimeLib
	if ortExplicit == "" {
		ortExplicit = os.Getenv("ONNXRUNTIME_LIB")
	}
	switch {
	case ortExplicit != "":
		ortPath = ortExplicit
		log.Printf("ORT: using explicit override %s", ortPath)
	case embedded.HasEmbeddedDylib():
		var err error
		ortTempPath, err = embedded.MaterializeDylib()
		if err != nil {
			log.Fatalf("ORT: failed to materialize embedded dylib: %v", err)
		}
		ortPath = ortTempPath
		log.Printf("ORT: using embedded dylib for %s (%d bytes) → %s",
			embedded.PlatformLabel(), embedded.OrtBytes(), ortPath)
	default:
		discovered := discoverOnnxRuntime()
		if discovered == "" {
			log.Fatal("Could not find libonnxruntime.{dylib,so}: no embedded dylib for this " +
				"build target (build with the right GOOS/GOARCH or supply -lib).")
		}
		ortPath = discovered
		log.Printf("ORT: no embedded dylib for %s/%s; using discovered system install %s",
			runtime.GOOS, runtime.GOARCH, discovered)
	}
	// Diagnostic: if both embedded AND a system install exist, warn on size
	// mismatch (likely different ORT versions, which can produce subtle bugs).
	if embedded.HasEmbeddedDylib() {
		if disc := discoverOnnxRuntime(); disc != "" {
			if info, err := os.Stat(disc); err == nil && info.Size() != int64(embedded.OrtBytes()) {
				log.Printf("WARN: embedded dylib is %d bytes; discovered local %s is %d bytes "+
					"(likely different ORT versions — using embedded)",
					embedded.OrtBytes(), disc, info.Size())
			}
		}
	}
	if ortTempPath != "" {
		defer os.Remove(ortTempPath)
	}

	if err := vad.InitONNXRuntime(ortPath); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer vad.DestroyONNXRuntime()

	// Resolve the backend's weights. Embedded-first; on-disk weights/<backend>
	// is the override path for backends added after the binary was built.
	backendName, ok := backendDirName(cfg.Model)
	if !ok {
		log.Fatalf("unknown VADModel %v", cfg.Model)
	}
	weightsDir := cfg.WeightsDir
	var weightsTempDir string
	if weightsDir == "" {
		var err error
		weightsDir, weightsTempDir, err = embedded.ResolveWeights(onDiskWeightsRoot, backendName)
		if err != nil {
			log.Fatalf("resolve weights for %s: %v", backendName, err)
		}
	}
	if weightsTempDir != "" {
		defer os.RemoveAll(weightsTempDir)
	}

	var (
		backend vad.Backend
		err     error
	)
	switch cfg.Model {
	case pb.VADModel_VAD_MODEL_PYANNOTE:
		backend, err = vad.NewModel(weightsDir)
	case pb.VADModel_VAD_MODEL_FSMN:
		backend, err = vad.NewFSMN(weightsDir)
	case pb.VADModel_VAD_MODEL_FIRERED:
		backend, err = vad.NewFireRed(weightsDir)
	case pb.VADModel_VAD_MODEL_SILERO:
		backend, err = vad.NewSilero(weightsDir)
	case pb.VADModel_VAD_MODEL_MARBLENET:
		backend, err = vad.NewMarbleNet(weightsDir)
	default:
		log.Fatalf("unknown VADModel %v", cfg.Model)
	}
	if err != nil {
		log.Fatalf("Failed to load %s backend: %v", cfg.Model.String(), err)
	}
	defer backend.Close()

	// Fetch RPC weights fetcher: any backend whose weights are embedded or on
	// disk can be served, regardless of which backend is loaded for inference.
	fetchBytes := func(m pb.VADModel) ([]byte, error) {
		name, ok := backendDirName(m)
		if !ok {
			return nil, fmt.Errorf("unknown model %v", m)
		}
		return embedded.WeightsBytes(onDiskWeightsRoot, name)
	}
	// Per-model URL lookup: returns the contents of url.txt (embedded-first,
	// disk fallback) so the Fetch RPC can redirect browser clients to a CDN
	// download instead of streaming bytes through gRPC.
	fetchURL := func(m pb.VADModel) (string, bool) {
		name, ok := backendDirName(m)
		if !ok {
			return "", false
		}
		return embedded.WeightsURL(onDiskWeightsRoot, name)
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Port))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", cfg.Port, err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxSendMsgSize(32*1024*1024),
		grpc.MaxRecvMsgSize(32*1024*1024),
	)
	pb.RegisterVoiceSegmentationServer(grpcServer,
		server.New(backend, cfg.Model, fetchBytes, fetchURL, cfg.WeightsUrl))
	reflection.Register(grpcServer)

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		grpcServer.GracefulStop()
	}()

	log.Printf("VAD gRPC server listening on :%d  (model=%s, weights=%s)",
		cfg.Port, cfg.Model.String(), filepath.Clean(weightsDir))
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
