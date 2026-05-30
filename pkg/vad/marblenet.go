package vad

import (
	"fmt"
	"math"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/accretional/vad/melspec"
)

// MarbleNet VAD model constants (from nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0
// export — see speax/benchmarks/vad/export_marblenet_to_onnx.py).
//
// Pipeline: log-Mel feats (NeMo conventions) → CNN encoder → 2-class softmax.
// Output is ONE logit pair per 20 ms of audio (T_out = T_mel / 2 because the
// encoder subsamples by 2). p_speech = softmax(logits)[..., 1].
const (
	marbleMels       = 80
	marbleFrameStepMs = 20 // each output frame represents 20 ms of audio
	marbleOnThreshold  = 0.5
	marbleOffThreshold = 0.3
	marbleMinOnFrames  = 10 // 200 ms
	marbleMinOffFrames = 5  // 100 ms
)

// MarbleNet is the NVIDIA MarbleNet VAD backend. Implements Backend.
//
// Loads ONNX from `<weightsDir>/model.onnx`. Preprocessing is fixed to NeMo's
// AudioToMelSpectrogramPreprocessor defaults (see melspec.NeMoDefaults).
type MarbleNet struct {
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
	mel     *melspec.MelSpec
}

// NewMarbleNet creates a new MarbleNet backend from a weights directory
// containing model.onnx.
func NewMarbleNet(weightsDir string) (*MarbleNet, error) {
	modelPath := filepath.Join(weightsDir, "model.onnx")
	session, err := ort.NewDynamicAdvancedSession(
		modelPath, []string{"audio_signal"}, []string{"outputs"}, nil)
	if err != nil {
		return nil, fmt.Errorf("create marblenet session: %w", err)
	}
	mel, err := melspec.New(melspec.NeMoDefaults())
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("create melspec: %w", err)
	}
	return &MarbleNet{session: session, mel: mel}, nil
}

// Close releases all resources.
func (m *MarbleNet) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
	}
}

// ProcessAudio runs MarbleNet VAD on raw 16 kHz mono float32 audio in [-1, 1].
func (m *MarbleNet) ProcessAudio(samples []float32) ([]Segment, error) {
	if len(samples) == 0 {
		return nil, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	// Compute NeMo-style log-mel features, transposed to channels-first
	// [1, 80, T_mel] for direct ONNX input.
	feats, tMel := m.mel.ComputeChannelsFirstFlat(samples)
	if tMel == 0 {
		return nil, nil
	}

	inShape := ort.NewShape(1, int64(marbleMels), int64(tMel))
	inputT, err := ort.NewTensor(inShape, feats)
	if err != nil {
		return nil, fmt.Errorf("create audio_signal tensor: %w", err)
	}
	defer inputT.Destroy()

	// Output: [1, T_out, 2] with T_out = T_mel / 2 (the encoder subsamples).
	tOut := tMel / 2
	outShape := ort.NewShape(1, int64(tOut), 2)
	outT, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, fmt.Errorf("create outputs tensor: %w", err)
	}
	defer outT.Destroy()

	if err := m.session.Run([]ort.Value{inputT}, []ort.Value{outT}); err != nil {
		return nil, fmt.Errorf("marblenet inference: %w", err)
	}

	// Softmax over the 2 logits per frame, take p_speech = softmax[..., 1].
	logits := outT.GetData()
	probs := make([]float32, tOut)
	for t := 0; t < tOut; t++ {
		a := float64(logits[t*2])
		b := float64(logits[t*2+1])
		// Numerically stable softmax: subtract max before exp.
		maxL := a
		if b > maxL {
			maxL = b
		}
		expA := math.Exp(a - maxL)
		expB := math.Exp(b - maxL)
		probs[t] = float32(expB / (expA + expB))
	}
	return marbleExtractSegments(probs), nil
}

// marbleExtractSegments runs the standard onset/offset hysteresis machine
// on per-20ms speech probs: trigger speech when prob crosses on_threshold
// (0.5), end when it drops below off_threshold (0.3). Min-on / min-off
// durations filter brief blips and very short gaps.
func marbleExtractSegments(probs []float32) []Segment {
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
		p := probs[i]
		if !inSpeech {
			if p >= float32(marbleOnThreshold) {
				speechRun++
				if speechRun >= marbleMinOnFrames {
					inSpeech = true
					start = i - speechRun + 1
					if start < 0 {
						start = 0
					}
				}
			} else {
				speechRun = 0
			}
		} else {
			if p < float32(marbleOffThreshold) {
				silenceRun++
				if silenceRun >= marbleMinOffFrames {
					segments = append(segments, Segment{
						Start:      marbleFrameToSec(start),
						End:        marbleFrameToSec(i - silenceRun + 1),
						SpeakerID:  0,
						Confidence: 1.0,
					})
					inSpeech = false
					silenceRun = 0
					speechRun = 0
				}
			} else {
				silenceRun = 0
			}
		}
	}
	if inSpeech {
		segments = append(segments, Segment{
			Start:      marbleFrameToSec(start),
			End:        marbleFrameToSec(n),
			SpeakerID:  0,
			Confidence: 1.0,
		})
	}
	return segments
}

func marbleFrameToSec(frame int) float64 {
	return float64(frame) * float64(marbleFrameStepMs) / 1000.0
}

// compile-time check.
var _ Backend = (*MarbleNet)(nil)
