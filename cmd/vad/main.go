package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/accretional/vad/internal/server"
	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
)

func main() {
	port := flag.Int("port", 50051, "gRPC server port")
	modelPath := flag.String("model", "weights/model.onnx", "path to ONNX model")
	libPath := flag.String("lib", "", "path to ONNX Runtime shared library (or set ONNXRUNTIME_LIB)")
	weightsURL := flag.String("weights-url", "", "URL to return from Fetch RPC instead of raw weights")
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

	model, err := vad.NewModel(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", *port, err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxSendMsgSize(32*1024*1024),
		grpc.MaxRecvMsgSize(32*1024*1024),
	)
	pb.RegisterVoiceSegmentationServer(grpcServer, server.New(model, *modelPath, *weightsURL))

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		grpcServer.GracefulStop()
	}()

	log.Printf("VAD gRPC server listening on :%d", *port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
