package main

import (
	"encoding/json"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ---------------------------------------------------------------------------
// Static asset sanity checks. The browser engine fans out into a bunch of
// modules (engine.js + worker.js + per-backend pipelines + dsp/) and a typo
// in the embed glob would silently drop one. These tests catch that.
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
	if !strings.Contains(html, `type="module"`) {
		t.Error("index.html missing <script type=\"module\"> (app.js is an ES module)")
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

func TestStaticFSContainsEngine(t *testing.T) {
	// app.js imports engine.js, engine.js imports worker.js + cache.js, worker
	// dynamically imports backends/<name>.js + dsp/*.js. Confirm the whole
	// tree shipped — embed.FS picks `static/*` so subdirectories need to
	// actually be present, not just declared.
	wantPaths := []string{
		"static/app.js",
		"static/js/engine.js",
		"static/js/worker.js",
		"static/js/cache.js",
		"static/js/dsp/fft.js",
		"static/js/dsp/fbank.js",
		"static/js/dsp/melspec.js",
		"static/js/backends/pyannote.js",
		"static/js/backends/fsmn.js",
		"static/js/backends/firered.js",
		"static/js/backends/silero.js",
		"static/js/backends/marblenet.js",
	}
	for _, p := range wantPaths {
		if _, err := staticFS.ReadFile(p); err != nil {
			t.Errorf("missing embedded asset %s: %v", p, err)
		}
	}
}

func TestStaticFSContainsSamples(t *testing.T) {
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
	// Loose lower bound: 3 ui files + 3 samples + 8+ JS modules.
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
	if count < 14 {
		t.Errorf("expected at least 14 static files (ui + samples + js modules), got %d", count)
	}
}

// ---------------------------------------------------------------------------
// /describe with no live gRPC backend — should still return JSON.
// ---------------------------------------------------------------------------

func TestHandleDescribeJSONShape(t *testing.T) {
	// Dial an unreachable address so reflection definitely fails — confirm
	// the handler still returns 200 + the fallback Methods list.
	conn, err := grpc.NewClient("localhost:1",
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	a := &app{
		vadAddr:   "localhost:1",
		vadClient: pb.NewVoiceSegmentationClient(conn),
		vadConn:   conn,
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
	if resp.VadAddr != "localhost:1" {
		t.Errorf("VadAddr = %q, want localhost:1", resp.VadAddr)
	}
	if len(resp.Methods) == 0 {
		t.Errorf("expected fallback Methods list")
	}
}

// ---------------------------------------------------------------------------
// /aux allowlist
// ---------------------------------------------------------------------------

func TestHandleAuxRejectsTraversal(t *testing.T) {
	a := &app{weightsRoot: ""}
	cases := []struct {
		path string
		want int
	}{
		{"/aux/", http.StatusBadRequest},                          // missing dir+file
		{"/aux/fsmn-vad/", http.StatusBadRequest},                 // missing file
		{"/aux/fsmn-vad/..%2Fmodel.onnx", http.StatusBadRequest},  // URL-decoded contains .. and /
		{"/aux/fsmn-vad/../model.onnx", http.StatusBadRequest},    // explicit traversal — http.ServeMux may rewrite; test handler directly
		{"/aux/unknown-dir/foo", http.StatusNotFound},             // not in allowlist
		{"/aux/fsmn-vad/not-allowed", http.StatusForbidden},       // file not in allowlist
	}
	for _, c := range cases {
		req := httptest.NewRequest(http.MethodGet, c.path, nil)
		rec := httptest.NewRecorder()
		a.handleAux(rec, req)
		if rec.Code != c.want {
			t.Errorf("path %s: status = %d, want %d (body: %s)", c.path, rec.Code, c.want, rec.Body.String())
		}
	}
}

func TestHandleAuxServesAllowedFile(t *testing.T) {
	// am.mvn is on the allowlist AND embedded in the weights tree. Confirm
	// the handler streams it back.
	a := &app{weightsRoot: ""}
	req := httptest.NewRequest(http.MethodGet, "/aux/fsmn-vad/am.mvn", nil)
	rec := httptest.NewRecorder()
	a.handleAux(rec, req)
	if rec.Code != http.StatusOK {
		t.Skipf("am.mvn not embedded in this build (run prep-embed.sh): status %d", rec.Code)
	}
	if rec.Body.Len() < 100 {
		t.Errorf("am.mvn served back as %d bytes, expected ≥100", rec.Body.Len())
	}
}
