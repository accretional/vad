package e2e_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	containerName = "vad-e2e-test"
	imageName     = "vad"
	containerPort = "50051"
)

var (
	client pb.VoiceSegmentationClient
	conn   *grpc.ClientConn
)

func TestMain(m *testing.M) {
	repoRoot := findRepoRoot()

	// Build the Docker image
	fmt.Println("Building Docker image...")
	build := exec.Command("docker", "build", "-t", imageName,
		"--build-arg", "MAIN_PKG=./cmd/vad", repoRoot)
	build.Stdout = os.Stdout
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Docker build failed: %v\n", err)
		os.Exit(1)
	}

	// Stop any previous container
	exec.Command("docker", "rm", "-f", containerName).Run()

	// Run the container with port mapping
	fmt.Println("Starting container...")
	run := exec.Command("docker", "run", "--rm", "-d",
		"--name", containerName,
		"-p", containerPort+":"+containerPort,
		imageName)
	out, err := run.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Docker run failed: %v\n%s\n", err, out)
		os.Exit(1)
	}

	// Wait for the server to be ready
	addr := "127.0.0.1:" + containerPort
	ready := false
	for i := 0; i < 30; i++ {
		time.Sleep(time.Second)
		c, err := grpc.NewClient(addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			continue
		}
		cl := pb.NewVoiceSegmentationClient(c)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, pingErr := cl.Detect(ctx, &pb.Audio{Samples: float32ToBytes(make([]float32, 16000)), SampleRate: 16000})
		cancel()
		if pingErr == nil {
			conn = c
			client = cl
			ready = true
			break
		}
		c.Close()
	}
	if !ready {
		cleanup()
		fmt.Fprintln(os.Stderr, "Failed to connect to container after 30s")
		os.Exit(1)
	}

	code := m.Run()

	conn.Close()
	cleanup()
	os.Exit(code)
}

func cleanup() {
	exec.Command("docker", "stop", containerName).Run()
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

func loadF32(path string) []byte {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(fmt.Sprintf("failed to load %s: %v", path, err))
	}
	return data
}

func float32ToBytes(samples []float32) []byte {
	buf := make([]byte, len(samples)*4)
	for i, s := range samples {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(s))
	}
	return buf
}

func TestDetectSilence(t *testing.T) {
	silence := make([]float32, 16000)
	resp, err := client.Detect(context.Background(), &pb.Audio{
		Samples:    float32ToBytes(silence),
		SampleRate: 16000,
	})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(resp.Segments) != 0 {
		t.Errorf("expected 0 segments for silence, got %d", len(resp.Segments))
	}
	if resp.Duration < 0.9 || resp.Duration > 1.1 {
		t.Errorf("expected duration ~1.0s, got %.3f", resp.Duration)
	}
}

func TestDetectSorryDave(t *testing.T) {
	repoRoot := findRepoRoot()
	data := loadF32(filepath.Join(repoRoot, "data", "sorry-dave-16k.f32"))

	resp, err := client.Detect(context.Background(), &pb.Audio{
		Samples:    data,
		SampleRate: 16000,
	})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(resp.Segments) < 2 {
		t.Errorf("expected ≥2 segments, got %d", len(resp.Segments))
	}
	if resp.Duration < 50 || resp.Duration > 60 {
		t.Errorf("expected duration ~53s, got %.3f", resp.Duration)
	}

	speakers := map[int32]bool{}
	for _, seg := range resp.Segments {
		speakers[seg.SpeakerId] = true
		if seg.Start >= seg.End {
			t.Errorf("segment has start >= end: %.3f >= %.3f", seg.Start, seg.End)
		}
		if seg.Confidence <= 0 || seg.Confidence > 1 {
			t.Errorf("segment confidence out of range: %.3f", seg.Confidence)
		}
	}
	if len(speakers) < 2 {
		t.Errorf("expected ≥2 speakers, got %d", len(speakers))
	}
}

func TestDetectWakeMeUp(t *testing.T) {
	repoRoot := findRepoRoot()
	data := loadF32(filepath.Join(repoRoot, "data", "wake-me-up-16k.f32"))

	resp, err := client.Detect(context.Background(), &pb.Audio{
		Samples:    data,
		SampleRate: 16000,
	})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(resp.Segments) != 0 {
		t.Errorf("expected 0 segments for music, got %d", len(resp.Segments))
	}
}

func TestDetectInvalidSampleRate(t *testing.T) {
	silence := make([]float32, 1000)
	_, err := client.Detect(context.Background(), &pb.Audio{
		Samples:    float32ToBytes(silence),
		SampleRate: 44100,
	})
	if err == nil {
		t.Fatal("expected error for wrong sample rate")
	}
	if !strings.Contains(err.Error(), "sample rate") {
		t.Errorf("error should mention sample rate, got: %v", err)
	}
}

func TestDetectEmptyAudio(t *testing.T) {
	_, err := client.Detect(context.Background(), &pb.Audio{
		Samples: nil,
	})
	if err == nil {
		t.Fatal("expected error for empty audio")
	}
}

func TestDetectMisalignedBytes(t *testing.T) {
	_, err := client.Detect(context.Background(), &pb.Audio{
		Samples: []byte{1, 2, 3},
	})
	if err == nil {
		t.Fatal("expected error for misaligned bytes")
	}
}
