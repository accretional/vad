// +build ignore

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
	if err := ort.InitializeEnvironment(); err != nil { log.Fatal(err) }
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
	if err != nil { log.Fatal(err) }
	defer session.Destroy()

	labels := []string{"∅", "A", "B", "A+B", "C", "A+C", "B+C"}

	// Process bestfriends - show ALL frames where non-silence is significant
	fmt.Println("=== bestfriends: frames where any speaker > 0.1 ===")
	samples := loadF32("data/bestfriends-16k.f32")
	copy(input.GetData(), samples)
	session.Run()
	data := output.GetData()

	for frame := 0; frame < 589; frame++ {
		probs := make([]float64, 7)
		for c := 0; c < 7; c++ {
			probs[c] = math.Exp(float64(data[frame*7+c]))
		}
		// Check if any non-silence class > 0.1
		hasSpeech := false
		for c := 1; c < 7; c++ {
			if probs[c] > 0.1 { hasSpeech = true }
		}
		if !hasSpeech { continue }

		t := float64(frame) * 10.0 / 589.0
		fmt.Printf("%.3fs: ", t)
		for c := 0; c < 7; c++ {
			if probs[c] > 0.01 {
				fmt.Printf("%s=%.3f ", labels[c], probs[c])
			}
		}
		fmt.Println()
	}

	// Process sorry-dave first 10s
	fmt.Println("\n=== sorry-dave (first 10s): frame transitions ===")
	samples2 := loadF32("data/sorry-dave-16k.f32")
	copy(input.GetData(), samples2[:160000])
	session.Run()
	data = output.GetData()

	prevClass := -1
	for frame := 0; frame < 589; frame++ {
		maxIdx := 0
		maxVal := float64(-999)
		for c := 0; c < 7; c++ {
			p := math.Exp(float64(data[frame*7+c]))
			if p > maxVal { maxVal = p; maxIdx = c }
		}
		if maxIdx != prevClass {
			t := float64(frame) * 10.0 / 589.0
			fmt.Printf("%.3fs: %s (p=%.3f)\n", t, labels[maxIdx], maxVal)
			prevClass = maxIdx
		}
	}
}
