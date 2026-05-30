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

	"github.com/accretional/vad/internal/server"
	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
)

// discoverOnnxRuntime looks for libonnxruntime.{dylib,so} in conventional
// locations so users don't have to pass -lib or set ONNXRUNTIME_LIB explicitly.
// Returns "" if nothing is found.
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
	// 1. Bundled under third_party/ relative to the executable (matches what
	//    setup.sh installs).
	if exe, err := os.Executable(); err == nil {
		for _, base := range []string{filepath.Dir(exe), filepath.Dir(filepath.Dir(exe))} {
			matches, _ := filepath.Glob(filepath.Join(base, "third_party", "onnxruntime-*", "lib", libName))
			candidates = append(candidates, matches...)
		}
	}
	// 2. Also try ./third_party from cwd (covers running from repo root).
	if matches, _ := filepath.Glob(filepath.Join("third_party", "onnxruntime-*", "lib", libName)); len(matches) > 0 {
		candidates = append(candidates, matches...)
	}
	// 3. System install locations.
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

// resolveModelPath picks the on-disk file to serve via the Fetch RPC. For
// pyannote, weights_dir is itself the .onnx file. For multi-file backends,
// the canonical entrypoint is `<weights_dir>/model.onnx`. Returns "" if the
// path doesn't resolve to an existing file (Fetch will then return an error).
func resolveModelPath(weightsDir string) string {
	if weightsDir == "" {
		return ""
	}
	st, err := os.Stat(weightsDir)
	if err != nil {
		return ""
	}
	if !st.IsDir() {
		return weightsDir
	}
	candidate := filepath.Join(weightsDir, "model.onnx")
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return ""
}

// defaultWeightsDir returns the on-disk path the server reaches for if
// VADConfig.weights_dir is unset.
func defaultWeightsDir(m pb.VADModel) string {
	switch m {
	case pb.VADModel_VAD_MODEL_PYANNOTE:
		return "weights/model.onnx"
	case pb.VADModel_VAD_MODEL_FSMN:
		return "weights/fsmn-vad"
	case pb.VADModel_VAD_MODEL_FIRERED:
		return "weights/firered-vad"
	case pb.VADModel_VAD_MODEL_MARBLENET:
		return "weights/marblenet"
	case pb.VADModel_VAD_MODEL_SILERO:
		return "weights/silero"
	}
	return ""
}

// legacyBackendName converts the deprecated -backend string into a VADModel.
// Returns VAD_MODEL_UNSPECIFIED for unknown / empty values (so the caller
// can preserve config-file or default behaviour).
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

func main() {
	configPath := flag.String("config", "",
		"path to a VADConfig textproto file (see proto/vad.proto for the schema)")
	backendStr := flag.String("backend", "",
		"DEPRECATED: legacy backend selector (pyannote | fsmn | firered | marblenet | silero). Prefer -config.")
	port := flag.Int("port", 0, "gRPC port (overrides config; default 50051)")
	modelPath := flag.String("model", "",
		"weights dir or ONNX path (overrides config; default per-model under weights/)")
	libPath := flag.String("lib", "", "ONNX Runtime shared library path (overrides ONNXRUNTIME_LIB env)")
	weightsURL := flag.String("weights-url", "",
		"URL returned by Fetch RPC; pyannote-only (overrides config)")
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

	// CLI flags override individual config fields.
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

	// Apply defaults.
	if cfg.Model == pb.VADModel_VAD_MODEL_UNSPECIFIED {
		cfg.Model = pb.VADModel_VAD_MODEL_PYANNOTE
	}
	if cfg.Port == 0 {
		cfg.Port = 50051
	}
	if cfg.OnnxruntimeLib == "" {
		cfg.OnnxruntimeLib = os.Getenv("ONNXRUNTIME_LIB")
	}
	if cfg.OnnxruntimeLib == "" {
		cfg.OnnxruntimeLib = discoverOnnxRuntime()
	}
	if cfg.OnnxruntimeLib == "" {
		log.Fatal("Could not find libonnxruntime.{dylib,so}. Pass -lib, set ONNXRUNTIME_LIB, " +
			"set onnxruntime_lib in -config, or run setup.sh to install it under third_party/.")
	}
	if cfg.WeightsDir == "" {
		cfg.WeightsDir = defaultWeightsDir(cfg.Model)
	}

	if err := vad.InitONNXRuntime(cfg.OnnxruntimeLib); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer vad.DestroyONNXRuntime()

	var (
		backend vad.Backend
		err     error
	)
	switch cfg.Model {
	case pb.VADModel_VAD_MODEL_PYANNOTE:
		backend, err = vad.NewModel(cfg.WeightsDir)
	case pb.VADModel_VAD_MODEL_FSMN:
		backend, err = vad.NewFSMN(cfg.WeightsDir)
	case pb.VADModel_VAD_MODEL_FIRERED:
		backend, err = vad.NewFireRed(cfg.WeightsDir)
	case pb.VADModel_VAD_MODEL_SILERO:
		backend, err = vad.NewSilero(cfg.WeightsDir)
	case pb.VADModel_VAD_MODEL_MARBLENET:
		log.Fatalf("backend %s not yet implemented (see TODO.md)", cfg.Model.String())
	default:
		log.Fatalf("unknown VADModel %v", cfg.Model)
	}
	if err != nil {
		log.Fatalf("Failed to load %s backend: %v", cfg.Model.String(), err)
	}
	defer backend.Close()

	// Fetch RPC: serve whatever ONNX file backs the loaded model. Generalized
	// from the original pyannote-only behaviour — any backend can have its
	// weights pulled by clients (e.g. transformers.js).
	modelFile := resolveModelPath(cfg.WeightsDir)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Port))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", cfg.Port, err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxSendMsgSize(32*1024*1024),
		grpc.MaxRecvMsgSize(32*1024*1024),
	)
	pb.RegisterVoiceSegmentationServer(grpcServer, server.New(backend, modelFile, cfg.WeightsUrl))
	reflection.Register(grpcServer)

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		grpcServer.GracefulStop()
	}()

	log.Printf("VAD gRPC server listening on :%d  (model=%s, weights=%s)",
		cfg.Port, cfg.Model.String(), filepath.Clean(cfg.WeightsDir))
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
