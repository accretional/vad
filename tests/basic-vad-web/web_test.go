package basicvadweb_test

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
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

	// Build the binary
	binPath := filepath.Join(repoRoot, "cmd", "basic-vad-web", "basic-vad-web")
	build := exec.Command("go", "build", "-o", binPath, "./cmd/basic-vad-web/")
	build.Dir = repoRoot
	build.Stdout = os.Stdout
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		panic("build failed: " + err.Error())
	}
	defer os.Remove(binPath)

	// Find ORT library
	libPath := findORTLib(repoRoot)
	if libPath == "" {
		panic("ONNX Runtime library not found")
	}

	// Start server
	serverCmd = exec.Command(binPath,
		"-port", serverPort,
		"-model", filepath.Join(repoRoot, "weights", "model.onnx"),
		"-lib", libPath,
	)
	serverCmd.Dir = repoRoot
	serverCmd.Stdout = os.Stdout
	serverCmd.Stderr = os.Stderr

	if runtime.GOOS == "darwin" {
		serverCmd.Env = append(os.Environ(),
			"DYLD_LIBRARY_PATH="+filepath.Dir(libPath))
	} else {
		serverCmd.Env = append(os.Environ(),
			"LD_LIBRARY_PATH="+filepath.Dir(libPath))
	}

	if err := serverCmd.Start(); err != nil {
		panic("failed to start server: " + err.Error())
	}

	// Wait for server to be ready
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

func findORTLib(repoRoot string) string {
	if p := os.Getenv("ONNXRUNTIME_LIB"); p != "" {
		return p
	}
	candidates := []string{
		"third_party/onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.dylib",
		"third_party/onnxruntime-osx-x86_64-1.22.0/lib/libonnxruntime.dylib",
		"third_party/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so",
		"third_party/onnxruntime-linux-aarch64-1.22.0/lib/libonnxruntime.so",
	}
	for _, c := range candidates {
		p := filepath.Join(repoRoot, c)
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
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
	if !strings.Contains(string(body), "/api/detect") {
		t.Error("app.js missing expected content")
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

func TestAPIDetectSilence(t *testing.T) {
	// Send 1s of silence as float32 PCM
	silence := make([]float32, 16000)
	buf := float32ToBytes(silence)

	resp, err := http.Post(
		"http://127.0.0.1:"+serverPort+"/api/detect",
		"application/octet-stream",
		strings.NewReader(string(buf)),
	)
	if err != nil {
		t.Fatalf("POST /api/detect failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}

	var result struct {
		Segments []struct {
			Start      float64 `json:"start"`
			End        float64 `json:"end"`
			SpeakerID  int     `json:"speaker_id"`
			Confidence float32 `json:"confidence"`
		} `json:"segments"`
		Duration float64 `json:"duration"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(result.Segments) != 0 {
		t.Errorf("expected 0 segments for silence, got %d", len(result.Segments))
	}
	if result.Duration < 0.9 || result.Duration > 1.1 {
		t.Errorf("expected duration ~1.0, got %.3f", result.Duration)
	}
}

func TestAPIDetectEmpty(t *testing.T) {
	resp, err := http.Post(
		"http://127.0.0.1:"+serverPort+"/api/detect",
		"application/octet-stream",
		strings.NewReader(""),
	)
	if err != nil {
		t.Fatalf("POST /api/detect failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 400 {
		t.Errorf("expected 400 for empty body, got %d", resp.StatusCode)
	}
}

func float32ToBytes(samples []float32) []byte {
	buf := make([]byte, len(samples)*4)
	for i, s := range samples {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(s))
	}
	return buf
}
