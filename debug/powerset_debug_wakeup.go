// +build ignore

// Detailed analysis of wake-me-up (music) audio file.
// Shows how the model handles non-speech audio: per-frame probability
// distributions, entropy, and class competition.

package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func loadF32(path string) []float32 {
	f, _ := os.Open(path)
	defer f.Close()
	data, _ := io.ReadAll(f)
	n := len(data) / 4
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		samples[i] = math.Float32frombits(bits)
	}
	return samples
}

func main() {
	libPath := os.Getenv("ONNXRUNTIME_LIB")
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()

	inputShape := ort.NewShape(1, 1, 160000)
	outputShape := ort.NewShape(1, 589, 7)
	input, _ := ort.NewEmptyTensor[float32](inputShape)
	output, _ := ort.NewEmptyTensor[float32](outputShape)
	defer input.Destroy()
	defer output.Destroy()
	session, err := ort.NewAdvancedSession("weights/model.onnx",
		[]string{"input_values"}, []string{"logits"},
		[]ort.Value{input}, []ort.Value{output}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	labels := []string{"∅", "A", "B", "A+B", "C", "A+C", "B+C"}

	samples := loadF32("data/wake-me-up-16k.f32")
	totalSamples := len(samples)
	duration := float64(totalSamples) / 16000.0
	numWindows := (totalSamples + 159999) / 160000

	fmt.Printf("=== wake-me-up: %d samples (%.2fs), %d window(s) ===\n",
		totalSamples, duration, numWindows)
	fmt.Println("Music file — expecting no speech segments.")
	fmt.Println()

	for w := 0; w < numWindows; w++ {
		offset := w * 160000
		windowStart := float64(offset) / 16000.0

		inputData := input.GetData()
		for i := range inputData {
			inputData[i] = 0
		}
		end := offset + 160000
		if end > totalSamples {
			end = totalSamples
		}
		copy(inputData, samples[offset:end])

		session.Run()
		data := output.GetData()

		fmt.Printf("--- Window %d (%.2f - %.2fs) ---\n",
			w, windowStart, windowStart+10.0)

		// Summary stats for this window
		silenceFrames := 0
		maxNonSilence := 0.0
		maxNonSilenceFrame := 0
		var sumEntropy float64

		for frame := 0; frame < 589; frame++ {
			probs := make([]float64, 7)
			for c := 0; c < 7; c++ {
				probs[c] = math.Exp(float64(data[frame*7+c]))
			}

			// Argmax
			maxIdx := 0
			maxVal := 0.0
			for c := 0; c < 7; c++ {
				if probs[c] > maxVal {
					maxVal = probs[c]
					maxIdx = c
				}
			}
			if maxIdx == 0 {
				silenceFrames++
			}

			// Max non-silence probability
			for c := 1; c < 7; c++ {
				if probs[c] > maxNonSilence {
					maxNonSilence = probs[c]
					maxNonSilenceFrame = frame
				}
			}

			// Shannon entropy
			var entropy float64
			for c := 0; c < 7; c++ {
				if probs[c] > 1e-10 {
					entropy -= probs[c] * math.Log2(probs[c])
				}
			}
			sumEntropy += entropy
		}

		fmt.Printf("  Silence frames: %d / 589 (%.1f%%)\n", silenceFrames, float64(silenceFrames)/589.0*100)
		fmt.Printf("  Max non-silence prob: %.4f (frame %d, t=%.3fs)\n",
			maxNonSilence, maxNonSilenceFrame,
			windowStart+float64(maxNonSilenceFrame)*10.0/589.0)
		fmt.Printf("  Mean entropy: %.3f bits (max possible: %.3f)\n",
			sumEntropy/589.0, math.Log2(7))
		fmt.Println()

		// Show every 20th frame for a sampled view
		fmt.Println("  Sampled frames (every 20th):")
		for frame := 0; frame < 589; frame += 20 {
			probs := make([]float64, 7)
			for c := 0; c < 7; c++ {
				probs[c] = math.Exp(float64(data[frame*7+c]))
			}
			t := windowStart + float64(frame)*10.0/589.0
			fmt.Printf("    %6.3fs: ", t)
			for c := 0; c < 7; c++ {
				fmt.Printf("%s=%.3f ", labels[c], probs[c])
			}
			fmt.Println()
		}

		// Show frames where non-silence is highest (top 5)
		fmt.Println()
		fmt.Println("  Top 5 frames with highest non-silence activity:")
		type frameInfo struct {
			idx    int
			maxP   float64
			maxC   int
			probs  [7]float64
		}
		var top5 [5]frameInfo
		for frame := 0; frame < 589; frame++ {
			probs := make([]float64, 7)
			for c := 0; c < 7; c++ {
				probs[c] = math.Exp(float64(data[frame*7+c]))
			}
			maxP := 0.0
			maxC := 0
			for c := 1; c < 7; c++ {
				if probs[c] > maxP {
					maxP = probs[c]
					maxC = c
				}
			}
			// Insert into top5 if bigger than smallest
			minIdx := 0
			for i := 1; i < 5; i++ {
				if top5[i].maxP < top5[minIdx].maxP {
					minIdx = i
				}
			}
			if maxP > top5[minIdx].maxP {
				top5[minIdx] = frameInfo{idx: frame, maxP: maxP, maxC: maxC}
				copy(top5[minIdx].probs[:], probs)
			}
		}
		for _, fi := range top5 {
			if fi.maxP == 0 {
				continue
			}
			t := windowStart + float64(fi.idx)*10.0/589.0
			fmt.Printf("    %6.3fs: best non-∅ = %s (%.4f) | ", t, labels[fi.maxC], fi.maxP)
			for c := 0; c < 7; c++ {
				fmt.Printf("%s=%.3f ", labels[c], fi.probs[c])
			}
			fmt.Println()
		}
		fmt.Println()
	}
}
