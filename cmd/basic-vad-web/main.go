// basic-vad-web: small Go HTTP server that fronts the vad gRPC service for a
// browser UI. All batch inference now runs in the browser via
// onnxruntime-web (see static/js/engine.js + worker.js); this server is the
// metadata + weights proxy that talks to ONE vad gRPC backend.
//
// HTTP / WS endpoints:
//
//   GET  /                serves static/index.html
//   GET  /static/...      embedded UI assets + samples
//   GET  /describe        JSON: service + RPC list + known backends
//   GET  /fetch?model=X   proxies VoiceSegmentation.Fetch — returns either
//                         {"url": "..."} (when the server has a CDN URL for
//                         that model, via url.txt sidecar) or the raw .onnx
//                         bytes as application/octet-stream
//   GET  /aux/<dir>/<f>   serves auxiliary files (am.mvn, cmvn_*.f32, ...)
//                         from the gRPC server's embedded weights tree.
//                         Implemented locally against the same embedded FS
//                         so we don't need a separate aux RPC.
//   WS   /socket          bridges browser → DetectStream against the loaded
//                         server-side backend (the live-streaming panel uses
//                         this; useful for testing server-side endpointing)
//
// No /detect anymore — the browser fans out to its own WebWorker-per-backend
// pipeline. Aux files are served from the same embedded weights tree the
// vad binary uses (internal/embedded.Weights) so this server doesn't need
// its own RPC to fetch them.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/accretional/vad/internal/embedded"
	pb "github.com/accretional/vad/proto/vadpb"
	"github.com/gorilla/websocket"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	reflectpb "google.golang.org/grpc/reflection/grpc_reflection_v1"
)

//go:embed static/*
var staticFS embed.FS

// allKnownModels is the proto-declared set of backends we can talk about.
// The browser renders one checkbox per entry. Every model in this list is
// supported by the browser-side inference engine (static/js/backends/);
// MarbleNet is no longer special-cased.
var allKnownModels = []pb.VADModel{
	pb.VADModel_VAD_MODEL_PYANNOTE,
	pb.VADModel_VAD_MODEL_FSMN,
	pb.VADModel_VAD_MODEL_FIRERED,
	pb.VADModel_VAD_MODEL_MARBLENET,
	pb.VADModel_VAD_MODEL_SILERO,
}

var modelDescriptions = map[pb.VADModel]string{
	pb.VADModel_VAD_MODEL_PYANNOTE:  "Pyannote Segmentation 3.0 — diarization (up to 3 speakers).",
	pb.VADModel_VAD_MODEL_FSMN:      "FunASR FSMN-VAD — tiny, Chinese-trained.",
	pb.VADModel_VAD_MODEL_FIRERED:   "FireRedTeam DFSMN-VAD — small, English-friendly.",
	pb.VADModel_VAD_MODEL_MARBLENET: "NVIDIA Frame_VAD_Multilingual_MarbleNet — multilingual.",
	pb.VADModel_VAD_MODEL_SILERO:    "Silero VAD — well-known tiny model.",
}

// backendDirName maps a VADModel to the directory name under weights/ that
// holds its onnx + aux files (mirrors cmd/vad/main.go).
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

// allowedAuxFiles enumerates the small sidecars each backend's browser-side
// pipeline reaches for via /aux/<dir>/<file>. Anything not on this list is
// rejected with 404 — we don't want to expose arbitrary file traversal.
var allowedAuxFiles = map[string]map[string]bool{
	"fsmn-vad":    {"am.mvn": true, "config.yaml": true},
	"firered-vad": {"cmvn_means.f32": true, "cmvn_istd.f32": true, "config.yaml": true, "cmvn.ark": true},
	"marblenet":   {"preprocessor.yaml": true},
}

// app is the runtime state passed to each handler.
type app struct {
	vadAddr      string
	vadClient    pb.VoiceSegmentationClient
	vadConn      *grpc.ClientConn
	defaultModel pb.VADModel
	weightsRoot  string // on-disk weights dir for fallback (usually "weights")

	// audio talks to the speax/audio MediaConverter gRPC service. Used by
	// /upload (decode arbitrary media → 16 kHz mono WAV) and /svg (waveform
	// SVG via AudioToVectors + VectorsToSvg). nil when -audio-addr was unset
	// — the corresponding endpoints then return 503.
	audio *audioApp
}

func main() {
	httpPort := flag.Int("port", 8080, "HTTP port for the web UI")
	vadAddr := flag.String("vad-addr", "localhost:50051", "host:port of the single vad gRPC backend")
	audioAddr := flag.String("audio-addr", "localhost:50052",
		"host:port of the speax/audio MediaConverter gRPC service. "+
			"Used for /upload (decode arbitrary media → 16 kHz mono WAV) and /svg "+
			"(waveform SVG via AudioToVectors + VectorsToSvg). "+
			"Set to empty to disable both endpoints.")
	weightsRoot := flag.String("weights-root", "weights",
		"on-disk weights/ directory used as fallback when /aux files aren't embedded")
	flag.Parse()

	conn, err := grpc.NewClient(*vadAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(64*1024*1024),
			grpc.MaxCallRecvMsgSize(64*1024*1024),
		),
	)
	if err != nil {
		log.Fatalf("dial vad %s: %v", *vadAddr, err)
	}
	defer conn.Close()

	a := &app{
		vadAddr:     *vadAddr,
		vadClient:   pb.NewVoiceSegmentationClient(conn),
		vadConn:     conn,
		weightsRoot: *weightsRoot,
	}

	// Audio backend is optional — without it /upload and /svg return 503,
	// but the rest of the demo (samples, in-browser inference, live mic)
	// keeps working.
	if *audioAddr != "" {
		au, err := newAudioApp(*audioAddr)
		if err != nil {
			log.Printf("audio backend at %s unavailable: %v (continuing without /upload + /svg)", *audioAddr, err)
		} else {
			a.audio = au
			if err := extractBundledSamples(); err != nil {
				log.Printf("extract bundled samples: %v", err)
			} else {
				go a.warmSampleSvgs()
			}
		}
	}
	// Try to learn the server's loaded model via reflection-free probe: call
	// Fetch with UNSPECIFIED, which makes the server use its defaultModel —
	// the response itself won't tell us which model, so we leave it blank in
	// /describe (the page falls back to "default for streaming") if we can't
	// reach the server during startup. Probing happens lazily in handleDescribe.

	mux := http.NewServeMux()

	staticSub, _ := fs.Sub(staticFS, "static")
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.FS(staticSub))))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		data, _ := staticFS.ReadFile("static/index.html")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(data)
	})

	mux.HandleFunc("/describe", a.handleDescribe)
	mux.HandleFunc("/fetch", a.handleFetch)
	mux.HandleFunc("/aux/", a.handleAux)
	mux.HandleFunc("/socket", a.handleSocket)
	mux.HandleFunc("/upload", a.handleUploadOr503)
	mux.HandleFunc("/svg", a.handleSvgOr503)

	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", *httpPort),
		Handler: mux,
	}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	log.Printf("basic-vad-web listening on http://localhost:%d (vad backend: %s)", *httpPort, *vadAddr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("ListenAndServe: %v", err)
	}
}

// ---------------------------------------------------------------------------
// /describe
// ---------------------------------------------------------------------------

type describeModel struct {
	Name        string `json:"name"`         // e.g. "VAD_MODEL_PYANNOTE"
	ShortName   string `json:"short_name"`   // e.g. "PYANNOTE"
	EnumValue   int    `json:"enum_value"`
	Description string `json:"description"`
}

type describeResponse struct {
	Service        string          `json:"service"`
	Methods        []string        `json:"methods"`
	Models         []describeModel `json:"models"`
	DefaultModel   string          `json:"default_model"`   // server's loaded backend (best-effort)
	VadAddr        string          `json:"vad_addr"`        // host:port of the gRPC server
	ReflectionNote string          `json:"reflection_note"` // empty if reflection worked
}

func (a *app) handleDescribe(w http.ResponseWriter, r *http.Request) {
	resp := describeResponse{
		Service: "vad.VoiceSegmentation",
		VadAddr: a.vadAddr,
	}
	for _, m := range allKnownModels {
		resp.Models = append(resp.Models, describeModel{
			Name:        m.String(),
			ShortName:   strings.TrimPrefix(m.String(), "VAD_MODEL_"),
			EnumValue:   int(m),
			Description: modelDescriptions[m],
		})
	}

	methods, reflErr := reflectMethods(r.Context(), a.vadConn)
	if reflErr != nil {
		resp.ReflectionNote = "reflection unavailable: " + reflErr.Error()
		resp.Methods = []string{"Detect", "DetectStream", "Fetch"}
	} else {
		sort.Strings(methods)
		resp.Methods = methods
	}

	// Best-effort: ask the server which model it has loaded. We only know if
	// it gives us a URL response and the URL ends with /<dir>/model.onnx, OR
	// if reflection were richer. For now leave default_model blank when we
	// can't tell — the page handles that gracefully.
	if loaded, ok := a.probeDefaultModel(r.Context()); ok {
		resp.DefaultModel = loaded.String()
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// probeDefaultModel asks the server to Fetch the UNSPECIFIED model — the
// server resolves that to its loaded default. We try to figure out which one
// it picked by matching the URL (when url.txt was used) against our backend
// directories. If we got bytes back, we can't tell; return (_, false).
func (a *app) probeDefaultModel(ctx context.Context) (pb.VADModel, bool) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	resp, err := a.vadClient.Fetch(ctx, &pb.FetchRequest{})
	if err != nil {
		return 0, false
	}
	url := resp.GetUrl()
	if url == "" {
		return 0, false
	}
	// Match URL suffix /<dir>/model.onnx against our backend dirs.
	for _, m := range allKnownModels {
		dir, ok := backendDirName(m)
		if !ok {
			continue
		}
		if strings.Contains(url, "/"+dir+"/") {
			return m, true
		}
	}
	return 0, false
}

func reflectMethods(ctx context.Context, conn *grpc.ClientConn) ([]string, error) {
	if conn == nil {
		return nil, errors.New("no vad connection")
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	rc := reflectpb.NewServerReflectionClient(conn)
	stream, err := rc.ServerReflectionInfo(ctx)
	if err != nil {
		return nil, err
	}
	if err := stream.Send(&reflectpb.ServerReflectionRequest{
		MessageRequest: &reflectpb.ServerReflectionRequest_FileContainingSymbol{
			FileContainingSymbol: "vad.VoiceSegmentation",
		},
	}); err != nil {
		return nil, err
	}
	resp, err := stream.Recv()
	if err != nil {
		return nil, err
	}
	fdr := resp.GetFileDescriptorResponse()
	if fdr == nil {
		return nil, fmt.Errorf("no file descriptor in response: %v", resp.GetMessageResponse())
	}
	if len(fdr.GetFileDescriptorProto()) == 0 {
		return nil, errors.New("empty file descriptor list")
	}
	return []string{"Detect", "DetectStream", "Fetch"}, nil
}

// ---------------------------------------------------------------------------
// /fetch — proxies the gRPC Fetch RPC
// ---------------------------------------------------------------------------

// handleFetch translates a browser GET /fetch?model=NAME into a Fetch RPC
// against the upstream vad server, then returns:
//
//   - application/json {"url": "..."} when the server has a URL for the model
//     (i.e. a url.txt sidecar is present). The browser then fetches the .onnx
//     directly from the CDN — much smaller round-trip through us.
//   - application/octet-stream raw bytes when the server returned weights.
//
// Either way the browser code in static/js/engine.js handles both shapes.
func (a *app) handleFetch(w http.ResponseWriter, r *http.Request) {
	q := strings.TrimSpace(strings.ToUpper(r.URL.Query().Get("model")))
	if q == "" {
		http.Error(w, "missing model= query param", http.StatusBadRequest)
		return
	}
	full := q
	if !strings.HasPrefix(full, "VAD_MODEL_") {
		full = "VAD_MODEL_" + full
	}
	v, ok := pb.VADModel_value[full]
	if !ok {
		http.Error(w, "unknown model "+q, http.StatusBadRequest)
		return
	}
	model := pb.VADModel(v)

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	resp, err := a.vadClient.Fetch(ctx, &pb.FetchRequest{Model: model})
	if err != nil {
		http.Error(w, "Fetch RPC failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	if url := resp.GetUrl(); url != "" {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"url": url})
		return
	}
	data := resp.GetWeights()
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
	w.Header().Set("Cache-Control", "public, max-age=86400")
	_, _ = w.Write(data)
}

// ---------------------------------------------------------------------------
// /aux/<dir>/<file>
// ---------------------------------------------------------------------------

// handleAux serves auxiliary files (am.mvn, cmvn_*.f32, ...) for backends
// whose browser-side pipeline needs more than just model.onnx. We use the
// SAME embedded weights tree the vad binary uses (internal/embedded.Weights),
// so we don't need an extra RPC.
func (a *app) handleAux(w http.ResponseWriter, r *http.Request) {
	// Path layout: /aux/<dir>/<file>. dir and file are both flat (no /).
	path := strings.TrimPrefix(r.URL.Path, "/aux/")
	parts := strings.SplitN(path, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		http.Error(w, "expected /aux/<backend_dir>/<file>", http.StatusBadRequest)
		return
	}
	dir, file := parts[0], parts[1]
	if strings.Contains(file, "/") || strings.Contains(file, "..") {
		http.Error(w, "no path components allowed in file", http.StatusBadRequest)
		return
	}
	allowed, ok := allowedAuxFiles[dir]
	if !ok {
		http.Error(w, "unknown backend dir "+dir, http.StatusNotFound)
		return
	}
	if !allowed[file] {
		http.Error(w, "file not in allowlist for "+dir, http.StatusForbidden)
		return
	}

	// Embedded-first.
	embedPath := "weights/" + dir + "/" + file
	if data, err := embedded.Weights.ReadFile(embedPath); err == nil {
		w.Header().Set("Content-Type", contentTypeForAux(file))
		w.Header().Set("Cache-Control", "public, max-age=86400")
		_, _ = w.Write(data)
		return
	}
	// Disk fallback.
	if a.weightsRoot != "" {
		candidate := filepath.Join(a.weightsRoot, dir, file)
		if data, err := os.ReadFile(candidate); err == nil {
			w.Header().Set("Content-Type", contentTypeForAux(file))
			_, _ = w.Write(data)
			return
		}
	}
	http.Error(w, "not found", http.StatusNotFound)
}

func contentTypeForAux(file string) string {
	switch {
	case strings.HasSuffix(file, ".yaml"):
		return "application/yaml"
	case strings.HasSuffix(file, ".mvn"):
		return "text/plain"
	default:
		return "application/octet-stream"
	}
}

// ---------------------------------------------------------------------------
// /socket  (WebSocket -> DetectStream bridge)
// ---------------------------------------------------------------------------

var wsUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// handleSocket bridges a browser WebSocket to the vad server's DetectStream
// RPC. Used by the live-streaming panel to test server-side endpointing
// against the loaded backend. The server picks the backend; there's no
// ?model= query anymore (the demo always uses the one vad backend the
// HTTP server is connected to).
func (a *app) handleSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	stream, err := a.vadClient.DetectStream(ctx)
	if err != nil {
		_ = conn.WriteJSON(map[string]any{"type": "error", "error": err.Error()})
		return
	}

	wsErrCh := make(chan error, 2)
	go func() {
		for {
			ev, err := stream.Recv()
			if err == io.EOF {
				wsErrCh <- nil
				return
			}
			if err != nil {
				wsErrCh <- err
				return
			}
			var payload any
			switch {
			case ev.GetActivity() != nil:
				payload = map[string]any{
					"type":          "activity",
					"speech_active": ev.GetActivity().GetSpeechActive(),
					"timestamp":     ev.GetTimestamp(),
				}
			case ev.GetSegment() != nil:
				s := ev.GetSegment()
				payload = map[string]any{
					"type":       "segment",
					"start":      s.GetStart(),
					"end":        s.GetEnd(),
					"speaker_id": s.GetSpeakerId(),
					"confidence": s.GetConfidence(),
					"timestamp":  ev.GetTimestamp(),
				}
			default:
				continue
			}
			if err := conn.WriteJSON(payload); err != nil {
				wsErrCh <- err
				return
			}
		}
	}()

	for {
		mt, data, err := conn.ReadMessage()
		if err != nil {
			_ = stream.Send(&pb.AudioChunk{EndOfStream: true, SampleRate: 16000})
			_ = stream.CloseSend()
			break
		}
		if mt == websocket.TextMessage {
			if strings.TrimSpace(string(data)) == "stop" {
				_ = stream.Send(&pb.AudioChunk{EndOfStream: true, SampleRate: 16000})
				_ = stream.CloseSend()
				break
			}
			continue
		}
		if mt != websocket.BinaryMessage {
			continue
		}
		if len(data)%4 != 0 {
			_ = conn.WriteJSON(map[string]any{
				"type":  "error",
				"error": "binary chunk not multiple of 4 bytes",
			})
			continue
		}
		if err := stream.Send(&pb.AudioChunk{
			Samples:    data,
			SampleRate: 16000,
		}); err != nil {
			_ = conn.WriteJSON(map[string]any{"type": "error", "error": err.Error()})
			break
		}
	}
	cancel()
	<-wsErrCh
}
