package main

import (
	"embed"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/accretional/vad/internal/server"
	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/grpc"
)

//go:embed static/*
var staticFS embed.FS

func main() {
	port := flag.Int("port", 8080, "server port (serves both gRPC and HTTP)")
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

	// gRPC server
	grpcServer := grpc.NewServer(
		grpc.MaxSendMsgSize(32*1024*1024),
		grpc.MaxRecvMsgSize(32*1024*1024),
	)
	pb.RegisterVoiceSegmentationServer(grpcServer, server.New(model, *modelPath, *weightsURL))

	// HTTP mux
	mux := http.NewServeMux()

	// Serve embedded static files
	staticSub, _ := fs.Sub(staticFS, "static")
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.FS(staticSub))))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		data, _ := staticFS.ReadFile("static/index.html")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(data)
	})

	// HTTP API endpoint for the web UI
	mux.HandleFunc("/api/detect", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(io.LimitReader(r.Body, 100*1024*1024)) // 100MB limit
		if err != nil {
			http.Error(w, "failed to read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		if len(body) == 0 {
			http.Error(w, "empty body", http.StatusBadRequest)
			return
		}
		if len(body)%4 != 0 {
			http.Error(w, "body length must be a multiple of 4 (float32 PCM)", http.StatusBadRequest)
			return
		}

		// Convert bytes to float32 samples
		n := len(body) / 4
		samples := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(body[i*4:])
			samples[i] = math.Float32frombits(bits)
		}

		segments, err := model.ProcessAudio(samples)
		if err != nil {
			http.Error(w, "inference failed: "+err.Error(), http.StatusInternalServerError)
			return
		}

		type segJSON struct {
			Start      float64 `json:"start"`
			End        float64 `json:"end"`
			SpeakerID  int     `json:"speaker_id"`
			Confidence float32 `json:"confidence"`
		}
		type respJSON struct {
			Segments []segJSON `json:"segments"`
			Duration float64   `json:"duration"`
		}

		resp := respJSON{
			Duration: float64(n) / float64(vad.SampleRate),
		}
		for _, s := range segments {
			resp.Segments = append(resp.Segments, segJSON{
				Start:      s.Start,
				End:        s.End,
				SpeakerID:  s.SpeakerID,
				Confidence: s.Confidence,
			})
		}
		if resp.Segments == nil {
			resp.Segments = []segJSON{}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// Dual handler: route gRPC vs HTTP based on content-type
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
			grpcServer.ServeHTTP(w, r)
		} else {
			mux.ServeHTTP(w, r)
		}
	})

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", *port, err)
	}

	httpServer := &http.Server{
		Handler: h2c.NewHandler(handler, &http2.Server{}),
	}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		httpServer.Close()
		grpcServer.GracefulStop()
	}()

	log.Printf("VAD web server listening on http://localhost:%d", *port)
	if err := httpServer.Serve(lis); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Failed to serve: %v", err)
	}
}
