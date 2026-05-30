// Integration tests for cmd/basic-vad-web. Builds the demo binary, runs it
// without a backing vad gRPC server (so /describe falls back to canned method
// list + /aux is satisfied from the embedded weights tree), and exercises
// the small set of HTTP endpoints the browser actually relies on:
//
//   GET /                    — index.html
//   GET /static/style.css    — CSS
//   GET /static/app.js       — main app module
//   GET /static/js/engine.js — engine module the app imports
//   GET /describe            — JSON metadata
//   GET /aux/fsmn-vad/am.mvn — embedded aux file
//   GET /aux/...../bogus     — 403 (not on allowlist)
//   GET /nonexistent         — 404
//
// Tests that need a live gRPC backend (Fetch RPC, /socket bridge) live in
// tests/e2e and tests/fetch.
package basicvadweb_test

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

const serverPort = "18080"

var serverCmd *exec.Cmd

func TestMain(m *testing.M) {
	repoRoot := findRepoRoot()

	binPath := filepath.Join(repoRoot, "cmd", "basic-vad-web", "basic-vad-web")
	build := exec.Command("go", "build", "-o", binPath, "./cmd/basic-vad-web/")
	build.Dir = repoRoot
	build.Stdout = os.Stdout
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		panic("build failed: " + err.Error())
	}
	defer os.Remove(binPath)

	// Run pointing at a deliberately unreachable vad addr; the endpoints
	// we test in this package don't need the gRPC backend up.
	serverCmd = exec.Command(binPath,
		"-port", serverPort,
		"-vad-addr", "127.0.0.1:1",
	)
	serverCmd.Dir = repoRoot
	serverCmd.Stdout = os.Stdout
	serverCmd.Stderr = os.Stderr

	if err := serverCmd.Start(); err != nil {
		panic("failed to start server: " + err.Error())
	}

	ready := false
	for i := 0; i < 30; i++ {
		time.Sleep(500 * time.Millisecond)
		resp, err := http.Get("http://127.0.0.1:" + serverPort + "/")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				ready = true
				break
			}
		}
	}
	if !ready {
		serverCmd.Process.Kill()
		panic("server did not become ready")
	}

	code := m.Run()

	serverCmd.Process.Kill()
	serverCmd.Wait()
	os.Exit(code)
}

func findRepoRoot() string {
	_, filename, _, _ := runtime.Caller(0)
	dir := filepath.Dir(filename)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

func TestIndexHTML(t *testing.T) {
	resp, err := http.Get("http://127.0.0.1:" + serverPort + "/")
	if err != nil {
		t.Fatalf("GET / failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/html") {
		t.Errorf("expected text/html content-type, got %q", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	html := string(body)
	if !strings.Contains(html, "Voice Activity Detection") {
		t.Error("index.html missing expected title")
	}
	if !strings.Contains(html, "style.css") {
		t.Error("index.html missing style.css link")
	}
	if !strings.Contains(html, "app.js") {
		t.Error("index.html missing app.js script")
	}
	if !strings.Contains(html, `type="module"`) {
		t.Error("index.html missing <script type=\"module\">")
	}
}

func TestStyleCSS(t *testing.T) {
	resp, err := http.Get("http://127.0.0.1:" + serverPort + "/static/style.css")
	if err != nil {
		t.Fatalf("GET /static/style.css failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "css") {
		t.Errorf("expected css content-type, got %q", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), ".container") {
		t.Error("style.css missing expected content")
	}
}

func TestAppJS(t *testing.T) {
	resp, err := http.Get("http://127.0.0.1:" + serverPort + "/static/app.js")
	if err != nil {
		t.Fatalf("GET /static/app.js failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "javascript") {
		t.Errorf("expected javascript content-type, got %q", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	// Sanity-check that app.js is the new ES module (imports engine).
	if !strings.Contains(string(body), "engine.js") {
		t.Error("app.js missing engine.js import (stale build?)")
	}
}

func TestEngineModuleServed(t *testing.T) {
	// engine.js / worker.js / per-backend modules all live under /static/js/.
	for _, path := range []string{
		"/static/js/engine.js",
		"/static/js/worker.js",
		"/static/js/cache.js",
		"/static/js/dsp/fft.js",
		"/static/js/backends/pyannote.js",
		"/static/js/backends/silero.js",
	} {
		resp, err := http.Get("http://127.0.0.1:" + serverPort + path)
		if err != nil {
			t.Errorf("GET %s failed: %v", path, err)
			continue
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Errorf("GET %s: status %d", path, resp.StatusCode)
		}
	}
}

func TestDescribe(t *testing.T) {
	resp, err := http.Get("http://127.0.0.1:" + serverPort + "/describe")
	if err != nil {
		t.Fatalf("GET /describe failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var got struct {
		Service string `json:"service"`
		Methods []string `json:"methods"`
		Models  []struct {
			Name      string `json:"name"`
			ShortName string `json:"short_name"`
		} `json:"models"`
		VadAddr string `json:"vad_addr"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Service != "vad.VoiceSegmentation" {
		t.Errorf("service = %q", got.Service)
	}
	if len(got.Models) < 5 {
		t.Errorf("expected ≥5 models, got %d", len(got.Models))
	}
	if got.VadAddr != "127.0.0.1:1" {
		t.Errorf("vad_addr = %q, want 127.0.0.1:1", got.VadAddr)
	}
	if len(got.Methods) == 0 {
		t.Errorf("expected non-empty methods (fallback list)")
	}
}

func TestAuxEmbeddedFile(t *testing.T) {
	// am.mvn should be embedded under internal/embedded/weights/fsmn-vad/.
	// The demo's /aux endpoint reads from that same embedded FS.
	resp, err := http.Get("http://127.0.0.1:" + serverPort + "/aux/fsmn-vad/am.mvn")
	if err != nil {
		t.Fatalf("GET /aux/fsmn-vad/am.mvn failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == 404 {
		t.Skip("am.mvn not embedded in this build (run prep-embed.sh)")
	}
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if len(body) < 100 {
		t.Errorf("am.mvn served back as %d bytes, expected ≥100", len(body))
	}
}

func TestAuxRejectsUnknownFile(t *testing.T) {
	resp, err := http.Get("http://127.0.0.1:" + serverPort + "/aux/fsmn-vad/secret.txt")
	if err != nil {
		t.Fatalf("GET aux unknown: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("expected 403 for non-allowlisted aux file, got %d", resp.StatusCode)
	}
}

func TestNotFoundReturns404(t *testing.T) {
	resp, err := http.Get("http://127.0.0.1:" + serverPort + "/nonexistent")
	if err != nil {
		t.Fatalf("GET /nonexistent failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("expected 404, got %d", resp.StatusCode)
	}
}
