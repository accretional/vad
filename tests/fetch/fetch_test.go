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
	imageName  = "vad"
	weightsURL = "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx"
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

// TestFetchWeightsDirect tests that Fetch returns raw model weights
// when no weights URL is configured.
func TestFetchWeightsDirect(t *testing.T) {
	client := startContainer(t, "vad-test-fetch-direct", "50052")

	resp, err := client.Fetch(context.Background(), &pb.FetchRequest{})
	if err != nil {
		t.Fatalf("Fetch failed: %v", err)
	}

	weights := resp.GetWeights()
	if weights == nil {
		t.Fatal("expected weights bytes, got URL or nil")
	}

	// model.onnx is ~5.99MB
	if len(weights) < 5_000_000 || len(weights) > 10_000_000 {
		t.Errorf("unexpected weights size: %d bytes", len(weights))
	}

	t.Logf("Received %d bytes of model weights", len(weights))
}

// TestFetchWeightsURL tests that Fetch returns a URL when the server
// is configured with -weights-url.
func TestFetchWeightsURL(t *testing.T) {
	client := startContainer(t, "vad-test-fetch-url", "50053", "-weights-url", weightsURL)

	resp, err := client.Fetch(context.Background(), &pb.FetchRequest{})
	if err != nil {
		t.Fatalf("Fetch failed: %v", err)
	}

	url := resp.GetUrl()
	if url == "" {
		t.Fatal("expected URL, got weights bytes or nil")
	}

	if url != weightsURL {
		t.Errorf("expected URL %q, got %q", weightsURL, url)
	}

	t.Logf("Received URL: %s", url)
}
