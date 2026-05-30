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

	"github.com/accretional/vad/internal/server"
	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

func main() {
	port := flag.Int("port", 50051, "gRPC server port")
	backend := flag.String("backend", "pyannote",
		"VAD backend: pyannote | fsmn | firered")
	modelPath := flag.String("model", "",
		"path to ONNX model (or directory for non-pyannote backends). "+
			"Default depends on -backend: weights/model.onnx for pyannote, "+
			"weights/fsmn-vad/ for fsmn, weights/firered-vad/ for firered.")
	libPath := flag.String("lib", "", "path to ONNX Runtime shared library (or set ONNXRUNTIME_LIB)")
	weightsURL := flag.String("weights-url", "", "URL to return from Fetch RPC (pyannote only)")
	flag.Parse()

	ortLib := *libPath
	if ortLib == "" {
		ortLib = os.Getenv("ONNXRUNTIME_LIB")
	}
	if ortLib == "" {
		log.Fatal("ONNX Runtime library path required: set -lib flag or ONNXRUNTIME_LIB env var")
	}

	if err := vad.InitONNXRuntime(ortLib); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer vad.DestroyONNXRuntime()

	// Per-backend defaults + load.
	var (
		model       vad.Backend
		pyannotePath string // path for the Fetch RPC (only set for pyannote)
		err          error
	)
	switch *backend {
	case "pyannote":
		path := *modelPath
		if path == "" {
			path = "weights/model.onnx"
		}
		pyannotePath = path
		model, err = vad.NewModel(path)
	case "fsmn":
		dir := *modelPath
		if dir == "" {
			dir = "weights/fsmn-vad"
		}
		model, err = vad.NewFSMN(dir)
	case "firered":
		dir := *modelPath
		if dir == "" {
			dir = "weights/firered-vad"
		}
		model, err = vad.NewFireRed(dir)
	default:
		log.Fatalf("unknown -backend %q (pyannote | fsmn | firered)", *backend)
	}
	if err != nil {
		log.Fatalf("Failed to load %s backend: %v", *backend, err)
	}
	defer model.Close()

	if *backend != "pyannote" && *weightsURL != "" {
		log.Printf("warning: -weights-url is currently only used by the pyannote backend's Fetch RPC")
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", *port, err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxSendMsgSize(32*1024*1024),
		grpc.MaxRecvMsgSize(32*1024*1024),
	)
	pb.RegisterVoiceSegmentationServer(grpcServer, server.New(model, pyannotePath, *weightsURL))
	// Reflection enables `grpcurl -plaintext localhost:50051 list` against the
	// live server, which is convenient for ad-hoc debugging.
	reflection.Register(grpcServer)

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		grpcServer.GracefulStop()
	}()

	resolved := *modelPath
	if resolved == "" {
		resolved = backendDefaultPath(*backend)
	}
	log.Printf("VAD gRPC server listening on :%d  (backend=%s, model=%s)",
		*port, *backend, filepath.Clean(resolved))
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

func backendDefaultPath(backend string) string {
	switch backend {
	case "pyannote":
		return "weights/model.onnx"
	case "fsmn":
		return "weights/fsmn-vad"
	case "firered":
		return "weights/firered-vad"
	}
	return ""
}
