package vad

import (
	"fmt"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Silero VAD model constants. The exported ONNX (see speax/benchmarks/vad/
// export_silero_to_onnx.py) operates on 32 ms (512-sample @ 16 kHz) chunks
// with explicit RNN state + context as input/output tensors — no fbank, no
// CMVN. Each chunk yields one speech probability.
const (
	sileroChunkSamples       = 512
	sileroChunkMs            = 32
	sileroStateDim0          = 2
	sileroStateDim1          = 128
	sileroContextDim         = 64
	sileroSpeechThreshold    = 0.5
	sileroMinSpeechFrames    = 4  // ~128 ms
	sileroMinSilenceFrames   = 10 // ~320 ms — close a segment after this much silence
)

// Silero is the snakers4/silero-vad backend. Implements Backend.
type Silero struct {
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
}

// NewSilero creates a new Silero VAD backend from a weights directory
// containing model.onnx.
func NewSilero(weightsDir string) (*Silero, error) {
	modelPath := filepath.Join(weightsDir, "model.onnx")
	inputs := []string{"input", "state", "context"}
	outputs := []string{"prob", "stateN", "contextN"}
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputs, outputs, nil)
	if err != nil {
		return nil, fmt.Errorf("create silero session: %w", err)
	}
	return &Silero{session: session}, nil
}

// Close releases all resources.
func (m *Silero) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
	}
}

// ProcessAudio runs Silero VAD on raw 16 kHz mono float32 audio in [-1, 1].
// State is reset at the start of each call (no continuity across calls — for
// streaming use DetectStream once a native streaming backend interface is
// added, see TODO.md).
func (m *Silero) ProcessAudio(samples []float32) ([]Segment, error) {
	if len(samples) == 0 {
		return nil, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	nChunks := len(samples) / sileroChunkSamples
	if nChunks == 0 {
		return nil, nil
	}

	// Pre-allocate fixed-shape tensors and reuse across chunks. State is
	// ping-ponged between two buffers (stateN → state next iteration).
	inShape := ort.NewShape(1, int64(sileroChunkSamples))
	stateShape := ort.NewShape(int64(sileroStateDim0), 1, int64(sileroStateDim1))
	ctxShape := ort.NewShape(1, int64(sileroContextDim))
	probShape := ort.NewShape(1, 1)

	inputT, err := ort.NewEmptyTensor[float32](inShape)
	if err != nil {
		return nil, fmt.Errorf("alloc input: %w", err)
	}
	defer inputT.Destroy()

	stateIn, err := ort.NewEmptyTensor[float32](stateShape)
	if err != nil {
		return nil, fmt.Errorf("alloc state: %w", err)
	}
	defer stateIn.Destroy()
	stateOut, err := ort.NewEmptyTensor[float32](stateShape)
	if err != nil {
		return nil, fmt.Errorf("alloc stateN: %w", err)
	}
	defer stateOut.Destroy()

	ctxIn, err := ort.NewEmptyTensor[float32](ctxShape)
	if err != nil {
		return nil, fmt.Errorf("alloc context: %w", err)
	}
	defer ctxIn.Destroy()
	ctxOut, err := ort.NewEmptyTensor[float32](ctxShape)
	if err != nil {
		return nil, fmt.Errorf("alloc contextN: %w", err)
	}
	defer ctxOut.Destroy()

	probT, err := ort.NewEmptyTensor[float32](probShape)
	if err != nil {
		return nil, fmt.Errorf("alloc prob: %w", err)
	}
	defer probT.Destroy()

	probs := make([]float32, nChunks)
	for i := 0; i < nChunks; i++ {
		copy(inputT.GetData(), samples[i*sileroChunkSamples:(i+1)*sileroChunkSamples])
		inputs := []ort.Value{inputT, stateIn, ctxIn}
		outputs := []ort.Value{probT, stateOut, ctxOut}
		if err := m.session.Run(inputs, outputs); err != nil {
			return nil, fmt.Errorf("silero chunk %d: %w", i, err)
		}
		probs[i] = probT.GetData()[0]
		// Ping-pong: copy stateN/contextN into stateIn/ctxIn for next chunk.
		copy(stateIn.GetData(), stateOut.GetData())
		copy(ctxIn.GetData(), ctxOut.GetData())
	}

	return sileroExtractSegments(probs), nil
}

// sileroExtractSegments runs a simple two-state hysteresis machine on the
// per-32ms speech probs.
func sileroExtractSegments(probs []float32) []Segment {
	n := len(probs)
	if n == 0 {
		return nil
	}
	var segments []Segment
	inSpeech := false
	start := 0
	speechRun := 0
	silenceRun := 0
	for i := 0; i < n; i++ {
		if probs[i] >= float32(sileroSpeechThreshold) {
			if !inSpeech {
				speechRun++
				if speechRun >= sileroMinSpeechFrames {
					inSpeech = true
					start = i - speechRun + 1
					if start < 0 {
						start = 0
					}
				}
			}
			silenceRun = 0
		} else {
			speechRun = 0
			if inSpeech {
				silenceRun++
				if silenceRun >= sileroMinSilenceFrames {
					segments = append(segments, Segment{
						Start:      sileroFrameToSec(start),
						End:        sileroFrameToSec(i - silenceRun + 1),
						SpeakerID:  0,
						Confidence: 1.0,
					})
					inSpeech = false
					silenceRun = 0
				}
			}
		}
	}
	if inSpeech {
		segments = append(segments, Segment{
			Start:      sileroFrameToSec(start),
			End:        sileroFrameToSec(n),
			SpeakerID:  0,
			Confidence: 1.0,
		})
	}
	return segments
}

func sileroFrameToSec(frame int) float64 {
	return float64(frame) * float64(sileroChunkMs) / 1000.0
}

// compile-time check.
var _ Backend = (*Silero)(nil)
