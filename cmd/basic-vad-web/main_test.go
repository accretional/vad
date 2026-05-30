package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ---------------------------------------------------------------------------
// Static asset sanity checks (carried over from the pre-multi-backend demo —
// kept so we don't silently ship an empty / broken index.html again).
// ---------------------------------------------------------------------------

func TestStaticFSContainsIndex(t *testing.T) {
	data, err := staticFS.ReadFile("static/index.html")
	if err != nil {
		t.Fatalf("failed to read static/index.html: %v", err)
	}
	html := string(data)
	if !strings.Contains(html, "<title>") {
		t.Error("index.html missing <title> tag")
	}
	if !strings.Contains(html, "Voice Activity Detection") {
		t.Error("index.html missing expected title text")
	}
	if !strings.Contains(html, "style.css") {
		t.Error("index.html missing style.css reference")
	}
	if !strings.Contains(html, "app.js") {
		t.Error("index.html missing app.js reference")
	}
}

func TestStaticFSContainsCSS(t *testing.T) {
	data, err := staticFS.ReadFile("static/style.css")
	if err != nil {
		t.Fatalf("failed to read static/style.css: %v", err)
	}
	if !strings.Contains(string(data), ".card") {
		t.Error("style.css missing .card rule")
	}
}

func TestStaticFSContainsJS(t *testing.T) {
	data, err := staticFS.ReadFile("static/app.js")
	if err != nil {
		t.Fatalf("failed to read static/app.js: %v", err)
	}
	js := string(data)
	if !strings.Contains(js, "/detect") {
		t.Error("app.js missing /detect endpoint")
	}
	if !strings.Contains(js, "/describe") {
		t.Error("app.js missing /describe endpoint")
	}
	if !strings.Contains(js, "/socket") {
		t.Error("app.js missing /socket WebSocket endpoint")
	}
}

func TestStaticFSContainsSamples(t *testing.T) {
	// Bundled samples must all be embedded so the page can fetch them.
	wantNames := []string{
		"static/samples/bestfriends.mp3",
		"static/samples/sorry-dave.mp3",
		"static/samples/wake-me-up.mp3",
	}
	for _, name := range wantNames {
		if _, err := staticFS.ReadFile(name); err != nil {
			t.Errorf("missing embedded sample %s: %v", name, err)
		}
	}
}

func TestStaticFSFileCount(t *testing.T) {
	// Loose lower bound: 3 ui files + 3 samples. Lets us add more files
	// without breaking the test, but catches the "embed wildcard isn't
	// picking up samples/" regression.
	var count int
	_ = fs.WalkDir(staticFS, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			count++
		}
		return nil
	})
	if count < 6 {
		t.Errorf("expected at least 6 static files (ui + samples), got %d", count)
	}
}

// ---------------------------------------------------------------------------
// parseVADAddrs
// ---------------------------------------------------------------------------

func TestParseVADAddrs(t *testing.T) {
	parsed, def, err := parseVADAddrs("PYANNOTE=localhost:50051,FSMN=localhost:50052,silero=h:1")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if def != pb.VADModel_VAD_MODEL_PYANNOTE {
		t.Errorf("default = %v, want PYANNOTE", def)
	}
	if parsed[pb.VADModel_VAD_MODEL_FSMN] != "localhost:50052" {
		t.Errorf("FSMN addr = %q, want localhost:50052", parsed[pb.VADModel_VAD_MODEL_FSMN])
	}
	if parsed[pb.VADModel_VAD_MODEL_SILERO] != "h:1" {
		t.Errorf("SILERO addr = %q, want h:1", parsed[pb.VADModel_VAD_MODEL_SILERO])
	}

	// Full-form keys.
	_, _, err = parseVADAddrs("VAD_MODEL_PYANNOTE=x:1")
	if err != nil {
		t.Errorf("full-form key should parse: %v", err)
	}

	// Bad inputs.
	if _, _, err := parseVADAddrs("PYANNOTE"); err == nil {
		t.Errorf("expected error on key without =")
	}
	if _, _, err := parseVADAddrs("BOGUS=x:1"); err == nil {
		t.Errorf("expected error on unknown model")
	}
}

// ---------------------------------------------------------------------------
// /describe with no live gRPC backend — should still return JSON.
// ---------------------------------------------------------------------------

func TestHandleDescribeJSONShape(t *testing.T) {
	// Dial an unreachable address so reflection definitely fails — we want to
	// confirm the handler still returns 200 + the fallback Methods list.
	conn, err := grpc.NewClient("localhost:1",
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	a := &app{
		clients: map[pb.VADModel]*backendClient{
			pb.VADModel_VAD_MODEL_PYANNOTE: {addr: "localhost:1", conn: conn, client: pb.NewVoiceSegmentationClient(conn)},
		},
		defaultModel: pb.VADModel_VAD_MODEL_PYANNOTE,
	}

	req := httptest.NewRequest(http.MethodGet, "/describe", nil)
	rec := httptest.NewRecorder()
	a.handleDescribe(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp describeResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Service != "vad.VoiceSegmentation" {
		t.Errorf("Service = %q", resp.Service)
	}
	if len(resp.Models) != len(allKnownModels) {
		t.Errorf("Models len = %d, want %d", len(resp.Models), len(allKnownModels))
	}
	// Pyannote should be the only available model.
	var available []string
	for _, m := range resp.Models {
		if m.Available {
			available = append(available, m.Name)
		}
	}
	if len(available) != 1 || available[0] != "VAD_MODEL_PYANNOTE" {
		t.Errorf("available = %v, want [VAD_MODEL_PYANNOTE]", available)
	}
	// Reflection failed -> fallback methods kicked in.
	if len(resp.Methods) == 0 {
		t.Errorf("expected fallback Methods list")
	}
}

// ---------------------------------------------------------------------------
// /detect — exercise the multipart + per-model fanout path using a mocked
// gRPC client. We swap in a stub by directly setting backendClient.client to
// a custom implementation.
// ---------------------------------------------------------------------------

type fakeVADClient struct {
	pb.VoiceSegmentationClient // embed for forward-compat
	detect                     func(ctx context.Context, in *pb.Audio, opts ...grpc.CallOption) (*pb.Diarization, error)
}

func (f *fakeVADClient) Detect(ctx context.Context, in *pb.Audio, opts ...grpc.CallOption) (*pb.Diarization, error) {
	return f.detect(ctx, in, opts...)
}

func TestHandleDetectMultipart(t *testing.T) {
	// Build the multipart body: 100 ms of silent f32 PCM at 16 kHz.
	const samples = 1600
	pcm := make([]byte, samples*4) // all zero -> silent
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, _ := mw.CreateFormFile("audio", "audio.f32le")
	_, _ = fw.Write(pcm)
	_ = mw.WriteField("encoding", "f32le")
	_ = mw.WriteField("model", "PYANNOTE")
	_ = mw.WriteField("model", "FSMN")
	mw.Close()

	// Wire up app with two fake backends.
	a := &app{
		clients: map[pb.VADModel]*backendClient{
			pb.VADModel_VAD_MODEL_PYANNOTE: {
				addr: "fake-pyannote",
				client: &fakeVADClient{detect: func(_ context.Context, in *pb.Audio, _ ...grpc.CallOption) (*pb.Diarization, error) {
					if len(in.Samples) != len(pcm) {
						t.Errorf("PYANNOTE got %d bytes, want %d", len(in.Samples), len(pcm))
					}
					return &pb.Diarization{
						Segments: []*pb.Segment{
							{Start: 0.0, End: 0.05, SpeakerId: 0, Confidence: 0.9},
						},
					}, nil
				}},
			},
			pb.VADModel_VAD_MODEL_FSMN: {
				addr: "fake-fsmn",
				client: &fakeVADClient{detect: func(_ context.Context, _ *pb.Audio, _ ...grpc.CallOption) (*pb.Diarization, error) {
					return &pb.Diarization{Segments: nil}, nil
				}},
			},
		},
		defaultModel: pb.VADModel_VAD_MODEL_PYANNOTE,
	}

	req := httptest.NewRequest(http.MethodPost, "/detect", body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	a.handleDetect(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp detectResponseJSON
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.AudioDurationSeconds < 0.099 || resp.AudioDurationSeconds > 0.101 {
		t.Errorf("AudioDurationSeconds = %f, want ~0.1", resp.AudioDurationSeconds)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("Results len = %d, want 2", len(resp.Results))
	}
	// Build a lookup so order doesn't matter.
	byShort := map[string]detectModelResultJSON{}
	for _, r := range resp.Results {
		byShort[r.ShortName] = r
	}
	if py, ok := byShort["PYANNOTE"]; !ok || len(py.Segments) != 1 {
		t.Errorf("expected PYANNOTE with 1 segment, got %+v", py)
	}
	if fsmn, ok := byShort["FSMN"]; !ok || len(fsmn.Segments) != 0 {
		t.Errorf("expected FSMN with 0 segments (empty list), got %+v", fsmn)
	}
}

func TestHandleDetectRejectsBadInputs(t *testing.T) {
	a := &app{
		clients:      map[pb.VADModel]*backendClient{},
		defaultModel: pb.VADModel_VAD_MODEL_PYANNOTE,
	}

	// Wrong method.
	req := httptest.NewRequest(http.MethodGet, "/detect", nil)
	rec := httptest.NewRecorder()
	a.handleDetect(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET status = %d, want 405", rec.Code)
	}

	// Missing model field.
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	fw, _ := mw.CreateFormFile("audio", "x.f32")
	_, _ = io.Copy(fw, bytes.NewReader(make([]byte, 4)))
	_ = mw.WriteField("encoding", "f32le")
	mw.Close()
	req = httptest.NewRequest(http.MethodPost, "/detect", body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec = httptest.NewRecorder()
	a.handleDetect(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("missing-model status = %d, want 400", rec.Code)
	}
}

