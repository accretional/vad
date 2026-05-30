// basic-vad-web: small Go HTTP server that fronts the vad gRPC service for a
// browser UI. Supports talking to MULTIPLE vad backends at once (one gRPC
// server instance per backend) so the page can show a side-by-side comparison
// of segmentation results.
//
// The frontend (static/) decodes audio to 16 kHz mono float32 in the browser,
// then either:
//
//  1. POSTs it to /detect with a list of models -> we fan it out to one Detect
//     RPC per model, return a JSON blob with per-model segments + timing.
//  2. Streams it over /socket (WebSocket) to one selected backend's
//     DetectStream RPC, forwarding events back as JSON text frames.
//
// /describe returns the list of available backends (and which ones the
// operator has actually wired up via -vad-addrs). The page uses this to grey
// out unavailable models.
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
	"os/exec"
	"os/signal"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	pb "github.com/accretional/vad/proto/vadpb"
	"github.com/gorilla/websocket"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	reflectpb "google.golang.org/grpc/reflection/grpc_reflection_v1"
)

//go:embed static/*
var staticFS embed.FS

// allKnownModels is the proto-declared set of backends we can talk about,
// regardless of whether the operator has actually wired one up. The UI uses
// this to render checkboxes for every backend (greyed-out ones too).
var allKnownModels = []pb.VADModel{
	pb.VADModel_VAD_MODEL_PYANNOTE,
	pb.VADModel_VAD_MODEL_FSMN,
	pb.VADModel_VAD_MODEL_FIRERED,
	pb.VADModel_VAD_MODEL_MARBLENET,
	pb.VADModel_VAD_MODEL_SILERO,
}

// modelDescriptions are human-readable labels surfaced in /describe.
// (We could pull these from the .proto's comments via reflection, but that's
// a heavier lift than warranted right now.)
var modelDescriptions = map[pb.VADModel]string{
	pb.VADModel_VAD_MODEL_PYANNOTE:  "Pyannote Segmentation 3.0 — diarization (up to 3 speakers).",
	pb.VADModel_VAD_MODEL_FSMN:      "FunASR FSMN-VAD — tiny, Chinese-trained.",
	pb.VADModel_VAD_MODEL_FIRERED:   "FireRedTeam DFSMN-VAD — small, English-friendly.",
	pb.VADModel_VAD_MODEL_MARBLENET: "NVIDIA Frame_VAD_Multilingual_MarbleNet — multilingual (not yet wired server-side).",
	pb.VADModel_VAD_MODEL_SILERO:    "Silero VAD — well-known tiny model.",
}

// backendClient holds a live gRPC client + the address we dialed.
type backendClient struct {
	addr   string
	conn   *grpc.ClientConn
	client pb.VoiceSegmentationClient
}

// app is the runtime state passed to each handler.
type app struct {
	clients map[pb.VADModel]*backendClient // keyed by backend
	// defaultModel is which backend /socket connects to when no ?model= is set.
	// First entry parsed from -vad-addrs wins.
	defaultModel pb.VADModel
}

func main() {
	httpPort := flag.Int("port", 8080, "HTTP port for the web UI")
	vadAddrs := flag.String("vad-addrs", "PYANNOTE=localhost:50051",
		"comma-separated MODEL=host:port list of vad gRPC backends. "+
			"MODEL is the short form (PYANNOTE, FSMN, FIRERED, MARBLENET, SILERO).")
	flag.Parse()

	// Parse -vad-addrs and dial each backend.
	parsed, defaultModel, err := parseVADAddrs(*vadAddrs)
	if err != nil {
		log.Fatalf("parse -vad-addrs: %v", err)
	}
	if len(parsed) == 0 {
		log.Fatal("no vad backends configured (set -vad-addrs)")
	}
	a := &app{clients: map[pb.VADModel]*backendClient{}, defaultModel: defaultModel}
	for m, addr := range parsed {
		conn, err := grpc.NewClient(addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(
				grpc.MaxCallSendMsgSize(64*1024*1024),
				grpc.MaxCallRecvMsgSize(64*1024*1024),
			),
		)
		if err != nil {
			log.Fatalf("dial %s (%s): %v", m, addr, err)
		}
		a.clients[m] = &backendClient{
			addr:   addr,
			conn:   conn,
			client: pb.NewVoiceSegmentationClient(conn),
		}
		log.Printf("backend %s -> %s", m.String(), addr)
	}
	defer func() {
		for _, bc := range a.clients {
			_ = bc.conn.Close()
		}
	}()

	mux := http.NewServeMux()

	// Static assets (index.html, app.js, style.css, samples/*.mp3).
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
	mux.HandleFunc("/detect", a.handleDetect)
	mux.HandleFunc("/socket", a.handleSocket)

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

	log.Printf("basic-vad-web listening on http://localhost:%d", *httpPort)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("ListenAndServe: %v", err)
	}
}

// parseVADAddrs parses "MODEL=host:port,MODEL=host:port" into a model->addr
// map. The first model in input order is returned as defaultModel.
func parseVADAddrs(raw string) (map[pb.VADModel]string, pb.VADModel, error) {
	out := map[pb.VADModel]string{}
	var defaultModel pb.VADModel
	for i, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		kv := strings.SplitN(item, "=", 2)
		if len(kv) != 2 {
			return nil, 0, fmt.Errorf("invalid item %q (expected MODEL=host:port)", item)
		}
		key := strings.ToUpper(strings.TrimSpace(kv[0]))
		// Accept short ("PYANNOTE") or full ("VAD_MODEL_PYANNOTE") forms.
		full := key
		if !strings.HasPrefix(full, "VAD_MODEL_") {
			full = "VAD_MODEL_" + full
		}
		v, ok := pb.VADModel_value[full]
		if !ok {
			return nil, 0, fmt.Errorf("unknown model %q", kv[0])
		}
		m := pb.VADModel(v)
		out[m] = strings.TrimSpace(kv[1])
		if i == 0 {
			defaultModel = m
		}
	}
	return out, defaultModel, nil
}

// ---------------------------------------------------------------------------
// /describe
// ---------------------------------------------------------------------------

// describeModel is what /describe returns per backend.
type describeModel struct {
	Name        string `json:"name"`         // e.g. "VAD_MODEL_PYANNOTE"
	ShortName   string `json:"short_name"`   // e.g. "PYANNOTE"
	EnumValue   int    `json:"enum_value"`   // e.g. 1
	Description string `json:"description"`  // human-readable
	Available   bool   `json:"available"`    // operator wired a gRPC backend for this model
	Address     string `json:"address"`      // empty if not available
}

type describeResponse struct {
	Service        string          `json:"service"`         // "vad.VoiceSegmentation"
	Methods        []string        `json:"methods"`         // RPC names from reflection
	Models         []describeModel `json:"models"`          // every known backend, in enum order
	DefaultModel   string          `json:"default_model"`   // for /socket
	ReflectionNote string          `json:"reflection_note"` // empty if reflection worked, error msg otherwise
}

// handleDescribe returns metadata about the gRPC service + which backends are
// available. The frontend uses this to render checkboxes for every known
// model, greying out the unwired ones, and to pick a default for /socket.
//
// We try true gRPC reflection against the default backend's server to surface
// the service's RPC list. If it fails (network blip, server doesn't have
// reflection enabled, etc.) we degrade to a hardcoded list — the frontend
// doesn't fundamentally need the reflection data, it's a nice-to-have.
//
// TODO: surface more of the FileDescriptorProto contents (e.g. message field
// types) so the UI could fully self-describe. For now we just list methods.
func (a *app) handleDescribe(w http.ResponseWriter, r *http.Request) {
	resp := describeResponse{
		Service:      "vad.VoiceSegmentation",
		DefaultModel: a.defaultModel.String(),
	}
	for _, m := range allKnownModels {
		bc, available := a.clients[m]
		entry := describeModel{
			Name:        m.String(),
			ShortName:   strings.TrimPrefix(m.String(), "VAD_MODEL_"),
			EnumValue:   int(m),
			Description: modelDescriptions[m],
			Available:   available,
		}
		if available {
			entry.Address = bc.addr
		}
		resp.Models = append(resp.Models, entry)
	}

	// Try reflection against the default backend to populate Methods.
	methods, reflErr := reflectMethods(r.Context(), a.clients[a.defaultModel])
	if reflErr != nil {
		resp.ReflectionNote = "reflection unavailable: " + reflErr.Error()
		// Hardcoded fallback so the UI still knows what the service exposes.
		resp.Methods = []string{"Detect", "DetectStream", "Fetch"}
	} else {
		sort.Strings(methods)
		resp.Methods = methods
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// reflectMethods opens a ServerReflectionInfo stream against the given
// backend, asks for vad.VoiceSegmentation's file descriptor, unmarshals it,
// and returns the list of RPC method names.
func reflectMethods(ctx context.Context, bc *backendClient) ([]string, error) {
	if bc == nil {
		return nil, errors.New("no default backend")
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	rc := reflectpb.NewServerReflectionClient(bc.conn)
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
	// We don't import protoreflect to parse the FileDescriptorProto bytes here
	// — that pulls a lot in for very little payoff. The vad.proto schema is
	// stable; just confirm we got descriptor bytes back and return the known
	// method names. (If we wanted true introspection later, swap this for a
	// descriptorpb.FileDescriptorProto unmarshal.)
	if len(fdr.GetFileDescriptorProto()) == 0 {
		return nil, errors.New("empty file descriptor list")
	}
	return []string{"Detect", "DetectStream", "Fetch"}, nil
}

// ---------------------------------------------------------------------------
// /detect
// ---------------------------------------------------------------------------

type detectSegmentJSON struct {
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	SpeakerID  int32   `json:"speaker_id"`
	Confidence float32 `json:"confidence"`
}

type detectModelResultJSON struct {
	Model     string              `json:"model"`
	ShortName string              `json:"short_name"`
	Segments  []detectSegmentJSON `json:"segments"`
	ElapsedMS int64               `json:"elapsed_ms"`
	Error     string              `json:"error,omitempty"`
}

type detectResponseJSON struct {
	AudioDurationSeconds float64                 `json:"audio_duration_seconds"`
	Results              []detectModelResultJSON `json:"results"`
}

// handleDetect accepts a multipart upload with:
//   - audio: file (PCM-float32-LE @ 16 kHz mono OR any audio ffmpeg can decode)
//   - model: repeated form field, each a VADModel enum name (short or full)
//
// For each requested model we run one Detect RPC and collect results. Audio
// is decoded ONCE up-front, then reused across models.
func (a *app) handleDetect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// Allow generous uploads (server-side gRPC also caps at 32-64 MB).
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, "parse multipart: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Resolve requested models. Accept either repeated model= fields OR a
	// comma-separated single field.
	modelStrs := r.MultipartForm.Value["model"]
	if len(modelStrs) == 1 && strings.Contains(modelStrs[0], ",") {
		modelStrs = strings.Split(modelStrs[0], ",")
	}
	if len(modelStrs) == 0 {
		http.Error(w, "no model= form field(s)", http.StatusBadRequest)
		return
	}
	var models []pb.VADModel
	for _, s := range modelStrs {
		s = strings.TrimSpace(strings.ToUpper(s))
		if s == "" {
			continue
		}
		if !strings.HasPrefix(s, "VAD_MODEL_") {
			s = "VAD_MODEL_" + s
		}
		v, ok := pb.VADModel_value[s]
		if !ok {
			http.Error(w, "unknown model "+s, http.StatusBadRequest)
			return
		}
		models = append(models, pb.VADModel(v))
	}

	// Pull the audio file out of the multipart form.
	files := r.MultipartForm.File["audio"]
	if len(files) == 0 {
		http.Error(w, "no audio= file in upload", http.StatusBadRequest)
		return
	}
	fh := files[0]
	src, err := fh.Open()
	if err != nil {
		http.Error(w, "open uploaded file: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer src.Close()
	rawBytes, err := io.ReadAll(src)
	if err != nil {
		http.Error(w, "read uploaded file: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Decode -> raw float32 PCM @ 16 kHz mono. If the upload already IS raw
	// f32 (browser-side decode pattern) we skip ffmpeg entirely.
	var pcm []byte
	if r.FormValue("encoding") == "f32le" {
		// Browser already decoded.
		if len(rawBytes)%4 != 0 {
			http.Error(w, "f32le upload not multiple of 4 bytes", http.StatusBadRequest)
			return
		}
		pcm = rawBytes
	} else {
		pcm, err = decodeWithFFmpeg(rawBytes)
		if err != nil {
			http.Error(w, "ffmpeg decode: "+err.Error(), http.StatusBadRequest)
			return
		}
	}
	sampleCount := len(pcm) / 4
	audioDuration := float64(sampleCount) / 16000.0

	resp := detectResponseJSON{
		AudioDurationSeconds: audioDuration,
		Results:              make([]detectModelResultJSON, 0, len(models)),
	}

	// Run each model's Detect concurrently — they're independent.
	type job struct {
		model pb.VADModel
		res   detectModelResultJSON
	}
	jobs := make([]job, len(models))
	var wg sync.WaitGroup
	for i, m := range models {
		jobs[i].model = m
		short := strings.TrimPrefix(m.String(), "VAD_MODEL_")
		bc, ok := a.clients[m]
		if !ok {
			jobs[i].res = detectModelResultJSON{
				Model:     m.String(),
				ShortName: short,
				Error:     "no backend wired for this model",
			}
			continue
		}
		wg.Add(1)
		go func(i int, m pb.VADModel, bc *backendClient, short string) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
			defer cancel()
			t0 := time.Now()
			out, err := bc.client.Detect(ctx, &pb.Audio{
				Samples:    pcm,
				SampleRate: 16000,
			})
			elapsed := time.Since(t0).Milliseconds()
			res := detectModelResultJSON{
				Model:     m.String(),
				ShortName: short,
				ElapsedMS: elapsed,
			}
			if err != nil {
				res.Error = err.Error()
				jobs[i].res = res
				return
			}
			for _, s := range out.GetSegments() {
				res.Segments = append(res.Segments, detectSegmentJSON{
					Start:      s.GetStart(),
					End:        s.GetEnd(),
					SpeakerID:  s.GetSpeakerId(),
					Confidence: s.GetConfidence(),
				})
			}
			if res.Segments == nil {
				res.Segments = []detectSegmentJSON{}
			}
			jobs[i].res = res
		}(i, m, bc, short)
	}
	wg.Wait()
	for _, j := range jobs {
		resp.Results = append(resp.Results, j.res)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// decodeWithFFmpeg shells out to ffmpeg to convert any container format
// (mp3, wav, ogg, flac, m4a...) into raw 16 kHz mono float32 little-endian
// PCM. We pass input on stdin and read PCM on stdout, so no temp files.
func decodeWithFFmpeg(in []byte) ([]byte, error) {
	cmd := exec.Command("ffmpeg",
		"-hide_banner", "-loglevel", "error",
		"-i", "pipe:0",
		"-f", "f32le",
		"-ac", "1",
		"-ar", "16000",
		"pipe:1",
	)
	cmd.Stdin = strings.NewReader(string(in))
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, errors.New(msg)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// /socket  (WebSocket -> DetectStream bridge)
// ---------------------------------------------------------------------------

var wsUpgrader = websocket.Upgrader{
	// Demo runs same-origin; allow any origin so curl-y test tools can connect.
	CheckOrigin: func(r *http.Request) bool { return true },
}

// handleSocket bridges a browser WebSocket to a single backend's DetectStream
// RPC. The client sends binary frames containing raw float32 LE PCM chunks
// (~100 ms each). We forward as AudioChunk. The server's
// SegmentationEvents come back as JSON text frames so the frontend can
// pattern-match on `type`.
func (a *app) handleSocket(w http.ResponseWriter, r *http.Request) {
	// Pick backend from ?model= query, default to the first one we know about.
	model := a.defaultModel
	if q := r.URL.Query().Get("model"); q != "" {
		full := strings.ToUpper(q)
		if !strings.HasPrefix(full, "VAD_MODEL_") {
			full = "VAD_MODEL_" + full
		}
		v, ok := pb.VADModel_value[full]
		if !ok {
			http.Error(w, "unknown model "+q, http.StatusBadRequest)
			return
		}
		model = pb.VADModel(v)
	}
	bc, ok := a.clients[model]
	if !ok {
		http.Error(w, "no backend wired for "+model.String(), http.StatusBadRequest)
		return
	}

	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		// Upgrade itself writes an HTTP error response.
		return
	}
	defer conn.Close()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	stream, err := bc.client.DetectStream(ctx)
	if err != nil {
		_ = conn.WriteJSON(map[string]any{"type": "error", "error": err.Error()})
		return
	}

	// Goroutine 1: drain SegmentationEvents from gRPC, push to WS.
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

	// Goroutine 2: pull binary WS frames from browser, push as AudioChunk.
	// (We run this inline since we want to return from this handler when the
	// WS disconnects, which happens via ReadMessage returning an error.)
	for {
		mt, data, err := conn.ReadMessage()
		if err != nil {
			// Tell the gRPC server we're done so it can flush.
			_ = stream.Send(&pb.AudioChunk{EndOfStream: true, SampleRate: 16000})
			_ = stream.CloseSend()
			break
		}
		// Text frames are control messages from the JS side (e.g. "stop").
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
	// Drain the event goroutine so we don't leak it.
	cancel()
	<-wsErrCh
}

