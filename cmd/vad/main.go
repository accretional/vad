package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"google.golang.org/protobuf/encoding/prototext"

	"github.com/accretional/vad/internal/server"
	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
)

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
		log.Fatal("ONNX Runtime library path required: set -lib flag, ONNXRUNTIME_LIB env, " +
			"or onnxruntime_lib in -config")
	}
	if cfg.WeightsDir == "" {
		cfg.WeightsDir = defaultWeightsDir(cfg.Model)
	}

	if err := vad.InitONNXRuntime(cfg.OnnxruntimeLib); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer vad.DestroyONNXRuntime()

	var (
		backend      vad.Backend
		pyannotePath string
		err          error
	)
	switch cfg.Model {
	case pb.VADModel_VAD_MODEL_PYANNOTE:
		pyannotePath = cfg.WeightsDir
		backend, err = vad.NewModel(cfg.WeightsDir)
	case pb.VADModel_VAD_MODEL_FSMN:
		backend, err = vad.NewFSMN(cfg.WeightsDir)
	case pb.VADModel_VAD_MODEL_FIRERED:
		backend, err = vad.NewFireRed(cfg.WeightsDir)
	case pb.VADModel_VAD_MODEL_MARBLENET, pb.VADModel_VAD_MODEL_SILERO:
		log.Fatalf("backend %s not yet implemented (see TODO.md)", cfg.Model.String())
	default:
		log.Fatalf("unknown VADModel %v", cfg.Model)
	}
	if err != nil {
		log.Fatalf("Failed to load %s backend: %v", cfg.Model.String(), err)
	}
	defer backend.Close()

	if cfg.Model != pb.VADModel_VAD_MODEL_PYANNOTE && cfg.WeightsUrl != "" {
		log.Printf("warning: weights_url is currently only used by the pyannote backend's Fetch RPC")
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Port))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", cfg.Port, err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxSendMsgSize(32*1024*1024),
		grpc.MaxRecvMsgSize(32*1024*1024),
	)
	pb.RegisterVoiceSegmentationServer(grpcServer, server.New(backend, pyannotePath, cfg.WeightsUrl))
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
