package vad

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// pyannote-specific powerset class mapping for segmentation 3.0:
// 3 speakers, max 2 active per frame → 7 classes.
//
//	class 0: ∅ (silence)
//	class 1: {A}
//	class 2: {B}
//	class 3: {A, B}
//	class 4: {C}
//	class 5: {A, C}
//	class 6: {B, C}
var powersetMapping = [pyannoteNumClasses][]int{
	0: {},     // silence
	1: {0},    // A
	2: {1},    // B
	3: {0, 1}, // A+B
	4: {2},    // C
	5: {0, 2}, // A+C
	6: {1, 2}, // B+C
}

const (
	pyannoteNumSpeakers = 3
	// WindowSize is 10 seconds of audio at 16 kHz — the pyannote-3.0 window length.
	WindowSize = 10 * SampleRate // 160000 samples
	// pyannoteNumFrames is the output frame count for one WindowSize input.
	pyannoteNumFrames = 589
	pyannoteNumClasses = 7
)

// Model is the Pyannote Segmentation 3.0 backend. Implements Backend.
//
// Kept as the type name `Model` (instead of `Pyannote`) for backward compat
// with the original vad/server.go API; cmd/vad's default backend constructs
// this via NewModel.
type Model struct {
	mu      sync.Mutex
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

// NewModel creates a new Pyannote VAD model from an ONNX file.
//
// Equivalent to NewPyannote — kept under the original name so older callers
// keep compiling without modification.
func NewModel(modelPath string) (*Model, error) {
	inputShape := ort.NewShape(1, 1, int64(WindowSize))
	outputShape := ort.NewShape(1, int64(pyannoteNumFrames), int64(pyannoteNumClasses))

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

	return &Model{session: session, input: input, output: output}, nil
}

// NewPyannote is an alias for NewModel for clarity at the call site once
// multiple backends exist.
func NewPyannote(modelPath string) (*Model, error) { return NewModel(modelPath) }

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

// ProcessAudio runs VAD on raw 16 kHz mono float32 audio samples. It processes
// the audio in 10-second windows and returns detected segments.
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
		segments := extractPyannoteSegments(m.output.GetData(), windowStart)
		allSegments = append(allSegments, segments...)
	}

	return mergePyannoteSegments(allSegments), nil
}

// extractPyannoteSegments converts log-softmax output into per-speaker voice
// segments using powerset decoding.
func extractPyannoteSegments(logits []float32, windowStartSec float64) []Segment {
	frameDuration := 10.0 / float64(pyannoteNumFrames)

	var segments []Segment

	type speakerState struct {
		active  bool
		start   float64
		sumConf float32
		count   int
	}
	states := make([]speakerState, pyannoteNumSpeakers)

	for frame := 0; frame < pyannoteNumFrames; frame++ {
		frameStart := windowStartSec + float64(frame)*frameDuration

		maxIdx := 0
		maxLogP := logits[frame*pyannoteNumClasses]
		for c := 1; c < pyannoteNumClasses; c++ {
			if logits[frame*pyannoteNumClasses+c] > maxLogP {
				maxLogP = logits[frame*pyannoteNumClasses+c]
				maxIdx = c
			}
		}
		confidence := float32(math.Exp(float64(maxLogP)))
		activeSpeakers := powersetMapping[maxIdx]

		for spk := 0; spk < pyannoteNumSpeakers; spk++ {
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

	endTime := windowStartSec + 10.0
	for spk := 0; spk < pyannoteNumSpeakers; spk++ {
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

// mergePyannoteSegments merges adjacent segments for the same speaker across
// window boundaries.
func mergePyannoteSegments(segments []Segment) []Segment {
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

// compile-time check that *Model satisfies Backend.
var _ Backend = (*Model)(nil)
