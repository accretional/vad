// validate connects to a running vad gRPC server and exercises the
// VoiceSegmentation RPCs that matter for a release-readiness check:
//
//	- Detect on 1 s of silence (the cheap ping)
//	- Fetch for every backend in the proto enum (proves all 5 backends'
//	  weights are reachable from this server, whether embedded or on disk)
//
// Exits 0 on success, non-zero on any check failure. Designed to be called
// repeatedly from release.sh against each built artifact (native binary,
// slim containers, fat-binary-in-debian) on distinct ports.
//
// Usage:
//
//	go run ./tests/validate -addr localhost:50051
//	go run ./tests/validate -addr localhost:50104 -wait 60s
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	addr := flag.String("addr", "localhost:50051", "vad gRPC server address (host:port)")
	wait := flag.Duration("wait", 30*time.Second, "max time to wait for server readiness")
	flag.Parse()

	conn, err := grpc.NewClient(*addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64*1024*1024)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "dial %s: %v\n", *addr, err)
		os.Exit(2)
	}
	defer conn.Close()
	client := pb.NewVoiceSegmentationClient(conn)

	fmt.Printf("=== validating %s (wait=%s) ===\n", *addr, *wait)

	if err := waitReady(client, *wait); err != nil {
		fmt.Fprintf(os.Stderr, "server not ready: %v\n", err)
		os.Exit(2)
	}

	fail := 0
	check := func(name string, f func() error) {
		t0 := time.Now()
		err := f()
		dt := time.Since(t0)
		if err != nil {
			fmt.Printf("  FAIL  %-30s %6dms  %v\n", name, dt.Milliseconds(), err)
			fail++
		} else {
			fmt.Printf("  OK    %-30s %6dms\n", name, dt.Milliseconds())
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	check("Detect/silence", func() error {
		silence := make([]byte, 16000*4) // 1 s of float32 zeros
		resp, err := client.Detect(ctx, &pb.Audio{Samples: silence, SampleRate: 16000})
		if err != nil {
			return err
		}
		if len(resp.Segments) != 0 {
			return fmt.Errorf("silence produced %d segments (want 0)", len(resp.Segments))
		}
		if resp.Duration < 0.9 || resp.Duration > 1.1 {
			return fmt.Errorf("duration %.3fs (want ~1.0s)", resp.Duration)
		}
		return nil
	})

	// One Fetch per backend. Each must return either a URL (sidecar
	// configured) or non-empty bytes (the raw .onnx). Either is fine —
	// what we're validating is that the weights are reachable at all.
	backends := []struct {
		name  string
		model pb.VADModel
	}{
		{"pyannote", pb.VADModel_VAD_MODEL_PYANNOTE},
		{"fsmn", pb.VADModel_VAD_MODEL_FSMN},
		{"firered", pb.VADModel_VAD_MODEL_FIRERED},
		{"silero", pb.VADModel_VAD_MODEL_SILERO},
		{"marblenet", pb.VADModel_VAD_MODEL_MARBLENET},
	}
	for _, b := range backends {
		b := b
		check("Fetch/"+b.name, func() error {
			resp, err := client.Fetch(ctx, &pb.FetchRequest{Model: b.model})
			if err != nil {
				return err
			}
			if url := resp.GetUrl(); url != "" {
				return nil
			}
			if w := resp.GetWeights(); len(w) > 0 {
				return nil
			}
			return fmt.Errorf("Fetch returned empty (no url, no weights)")
		})
	}

	fmt.Println()
	if fail > 0 {
		fmt.Printf("VALIDATION FAILED at %s: %d check(s) failed\n", *addr, fail)
		os.Exit(1)
	}
	fmt.Printf("VALIDATION OK at %s\n", *addr)
}

// waitReady polls Detect with silence until it succeeds or the deadline hits.
func waitReady(client pb.VoiceSegmentationClient, max time.Duration) error {
	deadline := time.Now().Add(max)
	silence := make([]byte, 16000*4)
	var lastErr error
	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err := client.Detect(ctx, &pb.Audio{Samples: silence, SampleRate: 16000})
		cancel()
		if err == nil {
			return nil
		}
		lastErr = err
		time.Sleep(time.Second)
	}
	return fmt.Errorf("timed out after %s: %v", max, lastErr)
}
