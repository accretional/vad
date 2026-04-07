package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/accretional/vad/internal/audio"
	"github.com/accretional/vad/pkg/vad"
)

func main() {
	modelPath := flag.String("model", "weights/model.onnx", "path to ONNX model")
	libPath := flag.String("lib", "", "path to ONNX Runtime shared library")
	dataDir := flag.String("data", "data", "directory containing -16k.f32 audio files")
	flag.Parse()

	if *libPath == "" {
		if envLib := os.Getenv("ONNXRUNTIME_LIB"); envLib != "" {
			*libPath = envLib
		} else {
			log.Fatal("Set -lib flag or ONNXRUNTIME_LIB env var to the onnxruntime shared library path")
		}
	}

	if err := vad.InitONNXRuntime(*libPath); err != nil {
		log.Fatalf("Init ONNX Runtime: %v", err)
	}
	defer vad.DestroyONNXRuntime()

	model, err := vad.NewModel(*modelPath)
	if err != nil {
		log.Fatalf("Load model: %v", err)
	}
	defer model.Close()

	// Find all -16k.f32 files
	entries, err := os.ReadDir(*dataDir)
	if err != nil {
		log.Fatalf("Read data dir: %v", err)
	}

	scriptDir := filepath.Dir(os.Args[0])
	if scriptDir == "" || scriptDir == "." {
		scriptDir, _ = os.Getwd()
		scriptDir = filepath.Join(scriptDir, "cmd", "pkg-example")
	}

	segDir := filepath.Join(scriptDir, "segmented-output")
	unsegDir := filepath.Join(scriptDir, "unsegmented-output")

	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, "-16k.f32") {
			continue
		}

		audioPath := filepath.Join(*dataDir, name)
		baseName := strings.TrimSuffix(name, "-16k.f32")

		fmt.Printf("\n=== Processing: %s ===\n", name)

		samples, err := audio.LoadF32(audioPath)
		if err != nil {
			log.Printf("  ERROR loading %s: %v", name, err)
			continue
		}

		fmt.Printf("  Loaded %d samples (%.2fs)\n",
			len(samples), float64(len(samples))/float64(vad.SampleRate))

		segments, err := model.ProcessAudio(samples)
		if err != nil {
			log.Printf("  ERROR processing %s: %v", name, err)
			continue
		}

		fmt.Printf("  Detected %d segments\n", len(segments))
		for _, seg := range segments {
			fmt.Printf("    [%.3f - %.3f] speaker_%d (conf: %.4f)\n",
				seg.Start, seg.End, seg.SpeakerID, seg.Confidence)
		}

		if len(segments) == 0 {
			fmt.Println("  No speech detected, skipping output generation.")
			continue
		}

		// Segmented output
		chunks := audio.Segmented(samples, segments)
		segBase := filepath.Join(segDir, baseName)
		if err := audio.SaveSegmented(segBase, chunks); err != nil {
			log.Printf("  ERROR saving segmented: %v", err)
		} else {
			fmt.Printf("  Saved %d segmented chunks to %s/\n", len(chunks), segDir)
		}

		// Unsegmented output
		streams := audio.Unsegmented(samples, segments)
		unsegBase := filepath.Join(unsegDir, baseName)
		if err := audio.SaveUnsegmented(unsegBase, streams); err != nil {
			log.Printf("  ERROR saving unsegmented: %v", err)
		} else {
			fmt.Printf("  Saved %d speaker streams to %s/\n", len(streams), unsegDir)
		}
	}

	fmt.Println("\nDone. Run encode-to-16k.sh --reverse on output dirs to convert to MP3.")
}
