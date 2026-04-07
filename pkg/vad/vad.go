package vad

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Segment represents a detected voice activity segment.
type Segment struct {
	Start      float64 // seconds
	End        float64 // seconds
	SpeakerID  int     // 0-indexed speaker (A=0, B=1, C=2)
	Confidence float32
}

// Powerset class mapping for pyannote segmentation 3.0:
// 3 speakers, max 2 active per frame → 7 classes.
//
//	class 0: ∅ (silence)
//	class 1: {A}
//	class 2: {B}
//	class 3: {A, B}
//	class 4: {C}
//	class 5: {A, C}
//	class 6: {B, C}
//
// Each entry lists which speakers (0=A, 1=B, 2=C) are active for that class.
var powersetMapping = [numClasses][]int{
	0: {},       // silence
	1: {0},      // A
	2: {1},      // B
	3: {0, 1},   // A+B
	4: {2},      // C
	5: {0, 2},   // A+C
	6: {1, 2},   // B+C
}

const numSpeakers = 3

// SampleRate is the expected audio sample rate (16kHz).
const SampleRate = 16000

// WindowSize is 10 seconds of audio at 16kHz (the model's expected input length).
const WindowSize = 10 * SampleRate // 160000 samples

// numFrames is the number of output frames for a 10-second window.
const numFrames = 589
const numClasses = 7

// Model holds an ONNX session for pyannote segmentation inference.
type Model struct {
	mu      sync.Mutex
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

// InitONNXRuntime initializes the ONNX Runtime with the shared library path.
// Must be called once before creating any Model instances.
func InitONNXRuntime(libPath string) error {
	ort.SetSharedLibraryPath(libPath)
	return ort.InitializeEnvironment()
}

// DestroyONNXRuntime cleans up the ONNX Runtime environment.
func DestroyONNXRuntime() error {
	return ort.DestroyEnvironment()
}

// NewModel creates a new VAD model from an ONNX file.
func NewModel(modelPath string) (*Model, error) {
	inputShape := ort.NewShape(1, 1, int64(WindowSize))
	outputShape := ort.NewShape(1, int64(numFrames), int64(numClasses))

	input, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		input.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"input_values"}, []string{"logits"},
		[]ort.Value{input}, []ort.Value{output},
		nil,
	)
	if err != nil {
		input.Destroy()
		output.Destroy()
		return nil, fmt.Errorf("create session: %w", err)
	}

	return &Model{
		session: session,
		input:   input,
		output:  output,
	}, nil
}

// Close releases all resources.
func (m *Model) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
	}
	if m.input != nil {
		m.input.Destroy()
	}
	if m.output != nil {
		m.output.Destroy()
	}
}

// ProcessAudio runs VAD on raw 16kHz mono float32 audio samples.
// It processes the audio in 10-second windows and returns detected segments.
func (m *Model) ProcessAudio(samples []float32) ([]Segment, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var allSegments []Segment

	for offset := 0; offset < len(samples); offset += WindowSize {
		inputData := m.input.GetData()

		for i := range inputData {
			inputData[i] = 0
		}

		copyEnd := offset + WindowSize
		if copyEnd > len(samples) {
			copyEnd = len(samples)
		}
		copy(inputData, samples[offset:copyEnd])

		if err := m.session.Run(); err != nil {
			return nil, fmt.Errorf("inference at offset %d: %w", offset, err)
		}

		windowStart := float64(offset) / float64(SampleRate)
		segments := extractSegments(m.output.GetData(), windowStart)
		allSegments = append(allSegments, segments...)
	}

	return mergeSegments(allSegments), nil
}

// extractSegments converts log-softmax output into per-speaker voice segments
// using powerset decoding. For each frame, the argmax class determines which
// speakers are active, and the class probability is used as confidence.
func extractSegments(logits []float32, windowStartSec float64) []Segment {
	frameDuration := 10.0 / float64(numFrames)

	var segments []Segment

	type speakerState struct {
		active  bool
		start   float64
		sumConf float32
		count   int
	}
	states := make([]speakerState, numSpeakers)

	for frame := 0; frame < numFrames; frame++ {
		frameStart := windowStartSec + float64(frame)*frameDuration

		// Find argmax class and its probability
		maxIdx := 0
		maxLogP := logits[frame*numClasses]
		for c := 1; c < numClasses; c++ {
			if logits[frame*numClasses+c] > maxLogP {
				maxLogP = logits[frame*numClasses+c]
				maxIdx = c
			}
		}
		confidence := float32(math.Exp(float64(maxLogP)))

		// Determine which speakers are active from the powerset mapping
		activeSpeakers := powersetMapping[maxIdx]

		// Update per-speaker state
		for spk := 0; spk < numSpeakers; spk++ {
			isActive := false
			for _, s := range activeSpeakers {
				if s == spk {
					isActive = true
					break
				}
			}

			if isActive {
				if !states[spk].active {
					states[spk].active = true
					states[spk].start = frameStart
					states[spk].sumConf = 0
					states[spk].count = 0
				}
				states[spk].sumConf += confidence
				states[spk].count++
			} else if states[spk].active {
				segments = append(segments, Segment{
					Start:      states[spk].start,
					End:        frameStart,
					SpeakerID:  spk,
					Confidence: states[spk].sumConf / float32(states[spk].count),
				})
				states[spk].active = false
			}
		}
	}

	// Close still-active segments
	endTime := windowStartSec + 10.0
	for spk := 0; spk < numSpeakers; spk++ {
		if states[spk].active {
			segments = append(segments, Segment{
				Start:      states[spk].start,
				End:        endTime,
				SpeakerID:  spk,
				Confidence: states[spk].sumConf / float32(states[spk].count),
			})
		}
	}

	return segments
}

// mergeSegments merges adjacent segments for the same speaker across window boundaries.
func mergeSegments(segments []Segment) []Segment {
	if len(segments) == 0 {
		return nil
	}

	var merged []Segment
	for _, seg := range segments {
		if len(merged) > 0 {
			last := &merged[len(merged)-1]
			if last.SpeakerID == seg.SpeakerID && seg.Start-last.End < 0.1 {
				last.End = seg.End
				continue
			}
		}
		merged = append(merged, seg)
	}
	return merged
}
