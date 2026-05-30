package vad

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/accretional/vad/fbank"
)

// FSMN-VAD model constants (from funasr/fsmn-vad export).
//
// The ONNX graph takes log-Mel-fbank features stacked into LFR groups of 5
// adjacent frames, followed by CMVN; output is a per-frame softmax over
// "phone-like" states where index 0 is the silence prob.
const (
	fsmnInputDim    = 400 // 80 mels × LFR m=5
	fsmnOutputDim   = 248 // softmax classes (silence at index 0)
	fsmnCacheChan   = 128
	fsmnCacheLen    = 19
	fsmnCacheDim    = 1
	fsmnNumCaches   = 4
	fsmnLFRM        = 5
	fsmnLFRN        = 1
	fsmnFrameStepMs = 10 // each output frame = 10 ms of audio
	// Silence-prob threshold: a frame is "speech" if silence_prob <= this.
	// FunASR's default speech_noise_thres is 0.6 in (speech-silence) space,
	// which since softmax sums to 1 is equivalent to silence_prob <= 0.2.
	fsmnSilenceProbThreshold = 0.2
	// Smoothing window: a frame is finalised as speech if >= fsmnSmoothMinSpeech
	// of the surrounding fsmnSmoothWindowFrames votes are speech.
	fsmnSmoothWindowFrames = 20
	fsmnSmoothMinSpeech    = 15
	// Max consecutive silence frames before closing a speech segment.
	fsmnSilenceHangoverFrames = 80 // 800 ms
)

// FSMN is the FunASR FSMN-VAD backend. Implements Backend.
//
// Loads the ONNX file from `<weightsDir>/model.onnx` and the CMVN normalization
// stats from `<weightsDir>/am.mvn`. Both files come from funasr/fsmn-vad on HF
// (the .onnx is our parity-validated export — see speax/benchmarks/vad/
// export_fsmn_vad_to_onnx.py).
type FSMN struct {
	mu          sync.Mutex
	session     *ort.DynamicAdvancedSession
	fb          *fbank.Fbank
	means       []float32 // [400] CMVN means (already negated in the file)
	rescales    []float32 // [400] CMVN rescales
	zeroCaches  []*ort.Tensor[float32]
}

// NewFSMN creates a new FSMN-VAD backend from a weights directory containing
// model.onnx and am.mvn.
func NewFSMN(weightsDir string) (*FSMN, error) {
	modelPath := filepath.Join(weightsDir, "model.onnx")
	mvnPath := filepath.Join(weightsDir, "am.mvn")

	means, rescales, err := parseFSMNMVN(mvnPath)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", mvnPath, err)
	}
	if len(means) != fsmnInputDim || len(rescales) != fsmnInputDim {
		return nil, fmt.Errorf("am.mvn dims: means=%d rescales=%d, want %d each",
			len(means), len(rescales), fsmnInputDim)
	}

	fbOpts := fbank.Defaults()
	fbOpts.WindowType = fbank.WindowHamming
	fbOpts.NumMelBins = 80
	fbOpts.FrameLengthMs = 25
	fbOpts.FrameShiftMs = 10
	fbOpts.PreemphCoeff = 0.97
	fbOpts.RemoveDCOffset = true
	fbOpts.SnipEdges = true
	fb, err := fbank.New(fbOpts)
	if err != nil {
		return nil, fmt.Errorf("build fbank: %w", err)
	}

	inputs := []string{"speech", "in_cache0", "in_cache1", "in_cache2", "in_cache3"}
	outputs := []string{"logits", "out_cache0", "out_cache1", "out_cache2", "out_cache3"}
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputs, outputs, nil)
	if err != nil {
		return nil, fmt.Errorf("create fsmn session: %w", err)
	}

	// Pre-create 4 zero cache tensors — they're identical across calls
	// (offline / one-shot inference uses zero state on every call).
	cacheShape := ort.NewShape(1, int64(fsmnCacheChan), int64(fsmnCacheLen), int64(fsmnCacheDim))
	zeroCaches := make([]*ort.Tensor[float32], fsmnNumCaches)
	for i := 0; i < fsmnNumCaches; i++ {
		t, err := ort.NewEmptyTensor[float32](cacheShape)
		if err != nil {
			for _, prev := range zeroCaches[:i] {
				prev.Destroy()
			}
			session.Destroy()
			return nil, fmt.Errorf("create cache tensor %d: %w", i, err)
		}
		zeroCaches[i] = t
	}

	return &FSMN{
		session:    session,
		fb:         fb,
		means:      means,
		rescales:   rescales,
		zeroCaches: zeroCaches,
	}, nil
}

// Close releases all resources.
func (m *FSMN) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
	}
	for _, t := range m.zeroCaches {
		if t != nil {
			t.Destroy()
		}
	}
}

// ProcessAudio runs FSMN-VAD on raw 16 kHz mono float32 audio in [-1, 1]
// (the same convention pyannote takes). Returns detected speech segments.
func (m *FSMN) ProcessAudio(samples []float32) ([]Segment, error) {
	if len(samples) == 0 {
		return nil, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	// 1. Scale [-1,1] float32 to int16 range. FunASR's WavFrontend multiplies
	//    by 32768 before kaldi.fbank; we match.
	scaled := make([]float32, len(samples))
	for i, s := range samples {
		scaled[i] = s * 32768.0
	}

	// 2. Compute fbank features [T, 80].
	feats := m.fb.Compute(scaled)
	if len(feats) == 0 {
		return nil, nil
	}

	// 3. LFR stack m=5, n=1 with edge replication for the tail.
	lfr := lfrStack(feats, fsmnLFRM, fsmnLFRN)
	if len(lfr) == 0 {
		return nil, nil
	}
	tLfr := len(lfr)

	// 4. Apply CMVN: x = (x + means) * rescales (means are already negated
	//    in am.mvn, so this is effectively (x - mean) * (1/std)).
	flat := make([]float32, tLfr*fsmnInputDim)
	for i := 0; i < tLfr; i++ {
		row := lfr[i]
		for j := 0; j < fsmnInputDim; j++ {
			flat[i*fsmnInputDim+j] = (row[j] + m.means[j]) * m.rescales[j]
		}
	}

	// 5. Build input tensor [1, T_lfr, 400].
	speechShape := ort.NewShape(1, int64(tLfr), int64(fsmnInputDim))
	speechIn, err := ort.NewTensor(speechShape, flat)
	if err != nil {
		return nil, fmt.Errorf("create speech tensor: %w", err)
	}
	defer speechIn.Destroy()

	// 6. Pre-allocate output tensors. Logits shape mirrors input time dim.
	logitsShape := ort.NewShape(1, int64(tLfr), int64(fsmnOutputDim))
	logitsOut, err := ort.NewEmptyTensor[float32](logitsShape)
	if err != nil {
		return nil, fmt.Errorf("create logits tensor: %w", err)
	}
	defer logitsOut.Destroy()

	cacheShape := ort.NewShape(1, int64(fsmnCacheChan), int64(fsmnCacheLen), int64(fsmnCacheDim))
	outCaches := make([]*ort.Tensor[float32], fsmnNumCaches)
	for i := range outCaches {
		t, err := ort.NewEmptyTensor[float32](cacheShape)
		if err != nil {
			for _, prev := range outCaches[:i] {
				if prev != nil {
					prev.Destroy()
				}
			}
			return nil, fmt.Errorf("create out_cache%d tensor: %w", i, err)
		}
		outCaches[i] = t
	}
	defer func() {
		for _, t := range outCaches {
			t.Destroy()
		}
	}()

	inputs := []ort.Value{
		speechIn,
		m.zeroCaches[0], m.zeroCaches[1], m.zeroCaches[2], m.zeroCaches[3],
	}
	outputs := []ort.Value{
		logitsOut,
		outCaches[0], outCaches[1], outCaches[2], outCaches[3],
	}
	if err := m.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("fsmn inference: %w", err)
	}

	// 7. State machine over silence probability.
	logits := logitsOut.GetData()
	silenceProb := make([]float32, tLfr)
	for i := 0; i < tLfr; i++ {
		silenceProb[i] = logits[i*fsmnOutputDim] // index 0 is silence
	}
	return fsmnExtractSegments(silenceProb), nil
}

// fsmnExtractSegments runs the FunASR-derived smoothing + hangover state
// machine over per-frame silence probabilities.
func fsmnExtractSegments(silenceProb []float32) []Segment {
	n := len(silenceProb)
	if n == 0 {
		return nil
	}
	// Per-frame raw speech label.
	raw := make([]bool, n)
	for i, p := range silenceProb {
		raw[i] = p <= float32(fsmnSilenceProbThreshold)
	}
	// Sliding-window smoothing.
	smoothed := make([]bool, n)
	half := fsmnSmoothWindowFrames / 2
	for i := 0; i < n; i++ {
		lo, hi := i-half, i+half
		if lo < 0 {
			lo = 0
		}
		if hi > n {
			hi = n
		}
		count := 0
		for j := lo; j < hi; j++ {
			if raw[j] {
				count++
			}
		}
		smoothed[i] = count >= fsmnSmoothMinSpeech
	}

	var segments []Segment
	inSpeech := false
	start := 0
	silenceRun := 0
	for i := 0; i < n; i++ {
		if smoothed[i] {
			if !inSpeech {
				inSpeech = true
				start = i
			}
			silenceRun = 0
		} else if inSpeech {
			silenceRun++
			if silenceRun >= fsmnSilenceHangoverFrames {
				segments = append(segments, Segment{
					Start:      framesToSec(start),
					End:        framesToSec(i - silenceRun + 1),
					SpeakerID:  0,
					Confidence: 1.0,
				})
				inSpeech = false
				silenceRun = 0
			}
		}
	}
	if inSpeech {
		segments = append(segments, Segment{
			Start:      framesToSec(start),
			End:        framesToSec(n),
			SpeakerID:  0,
			Confidence: 1.0,
		})
	}
	return segments
}

// framesToSec converts an LFR-frame index to seconds (10 ms per frame).
func framesToSec(frame int) float64 {
	return float64(frame) * float64(fsmnFrameStepMs) / 1000.0
}

// lfrStack groups every m consecutive D-dimensional frames into one
// (m*D)-dimensional frame, sliding by n frames between outputs. If the input
// is shorter than m frames, the last frame is replicated to reach length m
// (matches FunASR's WavFrontend behaviour).
func lfrStack(feats [][]float32, m, n int) [][]float32 {
	t := len(feats)
	if t == 0 {
		return nil
	}
	d := len(feats[0])

	if t < m {
		last := feats[t-1]
		for i := t; i < m; i++ {
			feats = append(feats, last)
		}
		t = m
	}
	tLfr := (t-m)/n + 1
	out := make([][]float32, tLfr)
	for i := 0; i < tLfr; i++ {
		row := make([]float32, m*d)
		for j := 0; j < m; j++ {
			copy(row[j*d:(j+1)*d], feats[i*n+j])
		}
		out[i] = row
	}
	return out
}

// parseFSMNMVN parses a FunASR-style am.mvn file. Layout (newline-sensitive):
//
//	<Nnet>
//	<Splice> 400 400
//	[ 0 ]
//	<AddShift> 400 400
//	<LearnRateCoef> 0 [ -8.31 ... -11.71 ]
//	<Rescale> 400 400
//	<LearnRateCoef> 0 [ 0.155 ... 0.1509654 ]
//	</Nnet>
//
// Returns (means, rescales) for CMVN application: x = (x + means) * rescales.
func parseFSMNMVN(path string) (means, rescales []float32, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var section string
	for _, line := range strings.Split(string(data), "\n") {
		trim := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trim, "<AddShift>"):
			section = "means"
		case strings.HasPrefix(trim, "<Rescale>"):
			section = "rescales"
		case strings.HasPrefix(trim, "<LearnRateCoef>") && section != "":
			lb := strings.Index(trim, "[")
			rb := strings.LastIndex(trim, "]")
			if lb < 0 || rb < 0 || rb <= lb {
				continue
			}
			fields := strings.Fields(trim[lb+1 : rb])
			vals := make([]float32, len(fields))
			for i, f := range fields {
				v, perr := strconv.ParseFloat(f, 32)
				if perr != nil {
					return nil, nil, fmt.Errorf("parse value %q: %w", f, perr)
				}
				vals[i] = float32(v)
			}
			if section == "means" {
				means = vals
			} else {
				rescales = vals
			}
			section = ""
		}
	}
	if means == nil || rescales == nil {
		return nil, nil, fmt.Errorf("missing <AddShift> or <Rescale> block")
	}
	return means, rescales, nil
}

// compile-time check.
var _ Backend = (*FSMN)(nil)
