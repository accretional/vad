package e2e_test

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// TestAllBackendsInContainer spins up the pre-built `vad` image once per
// backend and runs a single Detect against the reference clip. Validates
// that every backend's weights + the embedded ORT dylib actually work
// end-to-end inside the container (not just on the dev host).
//
// Skipped automatically if the `vad` image isn't present — test.sh's
// "Docker build" step is responsible for producing it.
func TestAllBackendsInContainer(t *testing.T) {
	if exec.Command("docker", "image", "inspect", imageName).Run() != nil {
		t.Skipf("docker image %q not found; run test.sh or `docker build -t %s .` first",
			imageName, imageName)
	}

	repoRoot := findRepoRoot()
	audio := loadF32(filepath.Join(repoRoot, "data", "sorry-dave-16k.f32"))
	audioSeconds := float64(len(audio)) / 4 / 16000

	cases := []struct {
		name   string
		flag   string
		port   string
		// loose sanity bounds — every backend should detect a meaningful
		// chunk of the 53 s clip without claiming the whole thing is speech.
		minTotalSec float64
		maxTotalSec float64
		minSegs     int
	}{
		{"pyannote", "pyannote", "55051", 20, 50, 4},
		{"fsmn", "fsmn", "55052", 20, 53, 1},
		{"firered", "firered", "55053", 20, 50, 4},
		{"silero", "silero", "55054", 20, 50, 4},
		{"marblenet", "marblenet", "55055", 20, 50, 4},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cn := fmt.Sprintf("vad-test-backend-%s", tc.flag)
			exec.Command("docker", "rm", "-f", cn).Run()

			run := exec.Command("docker", "run", "--rm", "-d",
				"--name", cn,
				"-p", tc.port+":50051",
				imageName,
				"-backend", tc.flag,
			)
			out, err := run.CombinedOutput()
			if err != nil {
				t.Fatalf("docker run %s: %v\n%s", tc.flag, err, out)
			}
			t.Cleanup(func() { exec.Command("docker", "rm", "-f", cn).Run() })

			addr := "127.0.0.1:" + tc.port
			var client pb.VoiceSegmentationClient
			var conn *grpc.ClientConn
			ready := false
			for i := 0; i < 30; i++ {
				time.Sleep(time.Second)
				c, derr := grpc.NewClient(addr,
					grpc.WithTransportCredentials(insecure.NewCredentials()))
				if derr != nil {
					continue
				}
				cl := pb.NewVoiceSegmentationClient(c)
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				// Ping with 1s of silence — every backend should respond
				// even if it returns zero segments.
				_, perr := cl.Detect(ctx, &pb.Audio{
					Samples:    float32ToBytes(make([]float32, 16000)),
					SampleRate: 16000,
				})
				cancel()
				if perr == nil {
					conn = c
					client = cl
					ready = true
					break
				}
				c.Close()
			}
			if !ready {
				logs, _ := exec.Command("docker", "logs", cn).CombinedOutput()
				t.Fatalf("%s never became ready; container logs:\n%s", tc.flag, logs)
			}
			defer conn.Close()

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			resp, err := client.Detect(ctx, &pb.Audio{
				Samples:    audio,
				SampleRate: 16000,
			})
			if err != nil {
				t.Fatalf("Detect failed: %v", err)
			}

			if resp.Duration < audioSeconds*0.95 || resp.Duration > audioSeconds*1.05 {
				t.Errorf("duration: got %.2fs, want ~%.2fs", resp.Duration, audioSeconds)
			}
			if len(resp.Segments) < tc.minSegs {
				t.Errorf("segments: got %d, want >= %d", len(resp.Segments), tc.minSegs)
			}
			var totalSpeech float64
			for _, s := range resp.Segments {
				if s.Start >= s.End {
					t.Errorf("bad segment %+v", s)
				}
				totalSpeech += s.End - s.Start
			}
			if totalSpeech < tc.minTotalSec || totalSpeech > tc.maxTotalSec {
				t.Errorf("total speech: got %.2fs, want %.2f..%.2fs",
					totalSpeech, tc.minTotalSec, tc.maxTotalSec)
			}
			t.Logf("%s: %d segments, %.2fs speech (%.1f%% of %.2fs)",
				tc.flag, len(resp.Segments), totalSpeech,
				100*totalSpeech/audioSeconds, audioSeconds)
		})
	}
}

