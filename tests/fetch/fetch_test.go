package fetch_test

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	imageName = "vad"

	// hostedURL is what we pass via the -weights-url flag for the
	// global-URL test. It only matters that it's a non-empty string the
	// server can echo back — we don't actually fetch from it.
	hostedURL = "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx"

	// sidecarURL is the URL the embedded weights/pyannote/url.txt sidecar
	// contains; the Fetch RPC prefers it over both raw bytes and the
	// -weights-url flag (see internal/server/server.go fetchURL precedence).
	sidecarURL = "https://raw.githubusercontent.com/accretional/vad/refs/heads/main/weights/pyannote/model.onnx"
)

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

func float32ToBytes(samples []float32) []byte {
	buf := make([]byte, len(samples)*4)
	for i, s := range samples {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(s))
	}
	return buf
}

func startContainer(t *testing.T, name, hostPort string, extraArgs ...string) pb.VoiceSegmentationClient {
	t.Helper()

	// Remove any previous container
	exec.Command("docker", "rm", "-f", name).Run()

	// Run pre-built container
	args := []string{"run", "--rm", "-d", "--name", name, "-p", hostPort + ":50051"}
	args = append(args, imageName)
	args = append(args, extraArgs...)
	run := exec.Command("docker", args...)
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("Docker run failed (did you run test.sh or build the image first?): %v\n%s", err, out)
	}

	t.Cleanup(func() {
		exec.Command("docker", "stop", name).Run()
	})

	// Wait for server to be ready
	addr := "127.0.0.1:" + hostPort
	for i := 0; i < 30; i++ {
		time.Sleep(time.Second)
		conn, err := grpc.NewClient(addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(32*1024*1024)))
		if err != nil {
			continue
		}
		cl := pb.NewVoiceSegmentationClient(conn)
		// Use Detect with silence as readiness check
		silence := make([]float32, 16000)
		silenceBytes := float32ToBytes(silence)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err = cl.Detect(ctx, &pb.Audio{Samples: silenceBytes, SampleRate: 16000})
		cancel()
		if err == nil {
			t.Cleanup(func() { conn.Close() })
			return cl
		}
		conn.Close()
	}
	t.Fatal("Failed to connect to container after 30s")
	return nil
}

// TestFetchWeightsSidecar — default container ships with embedded
// weights/<backend>/url.txt sidecars. The Fetch RPC must prefer those over
// streaming raw bytes.
func TestFetchWeightsSidecar(t *testing.T) {
	client := startContainer(t, "vad-test-fetch-sidecar", "50052")

	resp, err := client.Fetch(context.Background(), &pb.FetchRequest{})
	if err != nil {
		t.Fatalf("Fetch failed: %v", err)
	}
	url := resp.GetUrl()
	if url == "" {
		t.Fatalf("expected URL from url.txt sidecar, got weights bytes (%d) / nil",
			len(resp.GetWeights()))
	}
	if url != sidecarURL {
		t.Errorf("expected sidecar URL %q, got %q", sidecarURL, url)
	}
	t.Logf("Received sidecar URL: %s", url)
}

// TestFetchWeightsSidecarBeatsFlag — when both a -weights-url flag *and* a
// per-model url.txt sidecar are present, the sidecar wins (per the
// precedence comment in internal/server/server.go Fetch).
func TestFetchWeightsSidecarBeatsFlag(t *testing.T) {
	client := startContainer(t, "vad-test-fetch-flag", "50053", "-weights-url", hostedURL)

	resp, err := client.Fetch(context.Background(), &pb.FetchRequest{})
	if err != nil {
		t.Fatalf("Fetch failed: %v", err)
	}
	url := resp.GetUrl()
	if url == "" {
		t.Fatal("expected URL, got weights bytes or nil")
	}
	if url != sidecarURL {
		t.Errorf("sidecar should win over -weights-url; expected %q, got %q",
			sidecarURL, url)
	}
	t.Logf("Received URL (sidecar precedence over flag): %s", url)
}
