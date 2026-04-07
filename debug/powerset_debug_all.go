// +build ignore

// Processes ALL data/*-16k.f32 files, showing argmax transitions and
// multi-class detail (overlap / competing classes) for each window.

package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"

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

	files, _ := filepath.Glob("data/*-16k.f32")
	if len(files) == 0 {
		log.Fatal("No *-16k.f32 files found in data/")
	}

	for _, file := range files {
		samples := loadF32(file)
		totalSamples := len(samples)
		duration := float64(totalSamples) / 16000.0
		numWindows := (totalSamples + 159999) / 160000

		fmt.Printf("\n============================================================\n")
		fmt.Printf("=== %s: %d samples (%.2fs), %d window(s) ===\n",
			filepath.Base(file), totalSamples, duration, numWindows)
		fmt.Printf("============================================================\n")

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

			if numWindows > 1 {
				fmt.Printf("\n--- Window %d (%.2f - %.2fs) ---\n",
					w, windowStart, windowStart+10.0)
			}

			// Show transitions
			fmt.Println("\nArgmax transitions:")
			prevClass := -1
			for frame := 0; frame < 589; frame++ {
				maxIdx := 0
				maxVal := float64(-999)
				for c := 0; c < 7; c++ {
					p := math.Exp(float64(data[frame*7+c]))
					if p > maxVal {
						maxVal = p
						maxIdx = c
					}
				}
				if maxIdx != prevClass {
					t := windowStart + float64(frame)*10.0/589.0
					fmt.Printf("  %6.3fs: %-6s (p=%.3f)\n", t, labels[maxIdx], maxVal)
					prevClass = maxIdx
				}
			}

			// Show frames with interesting multi-speaker or overlap activity
			fmt.Println("\nMulti-class detail (overlap/competing classes):")
			printed := false
			for frame := 0; frame < 589; frame++ {
				probs := make([]float64, 7)
				for c := 0; c < 7; c++ {
					probs[c] = math.Exp(float64(data[frame*7+c]))
				}

				// Show if: any overlap class > 0.08, or 2+ non-silence > 0.05
				interesting := false
				for c := 3; c < 7; c++ {
					if probs[c] > 0.08 {
						interesting = true
					}
				}
				count := 0
				for c := 1; c < 7; c++ {
					if probs[c] > 0.05 {
						count++
					}
				}
				if count >= 2 && probs[0] < 0.85 {
					interesting = true
				}

				if !interesting {
					continue
				}

				t := windowStart + float64(frame)*10.0/589.0
				fmt.Printf("  %6.3fs: ", t)
				for c := 0; c < 7; c++ {
					if probs[c] > 0.01 {
						fmt.Printf("%s=%.3f ", labels[c], probs[c])
					}
				}
				fmt.Println()
				printed = true
			}
			if !printed {
				fmt.Println("  (none)")
			}
		}
	}
}
