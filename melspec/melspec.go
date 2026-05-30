// Package melspec computes log-Mel-spectrogram features in the
// torchaudio / NeMo convention, distinct from the Kaldi convention
// implemented in github.com/accretional/vad/fbank.
//
// The two packages overlap conceptually — both convert audio to log-Mel
// frames — but the details diverge in ways that matter for model parity:
//
//	             fbank (Kaldi style)         melspec (NeMo/torchaudio style)
//	window       Povey or Hamming (L-1)      Hann periodic=True (L)
//	stft         snip_edges=true             center=True, reflect padding
//	preemph      per-frame (after framing)   whole signal (before framing)
//	mel scale    HTK (1127·ln(1+f/700))      Slaney piecewise (norm=slaney)
//	fmin         20 Hz typical               0 Hz
//	log          log(max(mel, ε))            log(mel + ε), ε = 2^-24
//
// Using the wrong convention against a model produces "numerically plausible
// but model-broken" features — VAD treats real audio as noise. Parity vs
// torchaudio.transforms.MelSpectrogram is verified in melspec_test.go.
package melspec

import (
	"fmt"
	"math"
)

// Options configure a MelSpec instance. NeMoDefaults() returns the settings
// used by NeMo's AudioToMelSpectrogramPreprocessor for the MarbleNet VAD
// model; override fields when wiring up other models.
type Options struct {
	SampleRate    int
	WinLenSamples int     // window length in samples (typical 400 @ 16 kHz)
	HopLenSamples int     // STFT hop in samples (typical 160 @ 16 kHz)
	NFFT          int     // FFT size; must be >= WinLenSamples and power of 2
	NumMelBins    int     // typical 80
	FMinHz        float64 // typical 0 (NeMo) or 20 (Kaldi)
	FMaxHz        float64 // 0 means SampleRate/2
	PreemphCoeff  float64 // 0.97 typical; set to 0 to disable
	LogOffset     float64 // 2^-24 typical (NeMo log_zero_guard_value); set to 0 for no offset
	PadToMultiple int     // right-pad time axis with zeros to this multiple; 0 disables
}

// NeMoDefaults returns the exact configuration used by NVIDIA NeMo's
// AudioToMelSpectrogramPreprocessor for the MarbleNet VAD model at 16 kHz.
// These values match the preprocessor.yaml saved alongside the ONNX export.
func NeMoDefaults() Options {
	return Options{
		SampleRate:    16000,
		WinLenSamples: 400, // 25 ms
		HopLenSamples: 160, // 10 ms
		NFFT:          512,
		NumMelBins:    80,
		FMinHz:        0,
		FMaxHz:        0, // ⇒ 8000
		PreemphCoeff:  0.97,
		LogOffset:     math.Ldexp(1, -24), // 2^-24
		PadToMultiple: 2,
	}
}

// MelSpec computes log-Mel features. Construct once (window + filterbank are
// precomputed); Compute is safe to call repeatedly. Not safe for concurrent
// calls on the same instance.
type MelSpec struct {
	opts    Options
	window  []float64 // Hann window of length WinLenSamples
	mel     [][]float64 // [NumMelBins][fftBins] Slaney filterbank
	fftBins int       // NFFT/2 + 1

	// Per-call scratch space reused across frames.
	frame  []float64 // length NFFT
	fftBuf []complex128
}

// New constructs a MelSpec with the given options.
func New(opts Options) (*MelSpec, error) {
	if opts.SampleRate <= 0 || opts.WinLenSamples <= 0 || opts.HopLenSamples <= 0 {
		return nil, fmt.Errorf("melspec: SampleRate, WinLenSamples, HopLenSamples must be > 0")
	}
	if opts.NumMelBins <= 0 {
		return nil, fmt.Errorf("melspec: NumMelBins must be > 0")
	}
	if opts.NFFT < opts.WinLenSamples || (opts.NFFT&(opts.NFFT-1)) != 0 {
		return nil, fmt.Errorf("melspec: NFFT must be a power of 2 >= WinLenSamples")
	}
	fmax := opts.FMaxHz
	if fmax <= 0 {
		fmax = float64(opts.SampleRate) / 2
	}
	fftBins := opts.NFFT/2 + 1
	return &MelSpec{
		opts:    opts,
		window:  hannPeriodic(opts.WinLenSamples),
		mel:     makeMelFilterbankSlaney(opts.NumMelBins, fftBins, float64(opts.SampleRate), opts.FMinHz, fmax),
		fftBins: fftBins,
		frame:   make([]float64, opts.NFFT),
		fftBuf:  make([]complex128, opts.NFFT),
	}, nil
}

// Compute returns log-mel-spectrogram features for the given mono float32
// PCM samples in [-1, 1] (the same convention models expect at the API
// boundary). Output shape is `[T_mel][NumMelBins]` — row-major, time-first.
//
// To pass to a NeMo-style ONNX model expecting `[1, NumMelBins, T_mel]`
// (channels-first), the caller transposes — see ComputeChannelsFirstFlat.
func (m *MelSpec) Compute(samples []float32) [][]float32 {
	tMel, padded := m.computeImpl(samples)
	if tMel == 0 {
		return nil
	}
	out := make([][]float32, tMel)
	for t := 0; t < tMel; t++ {
		out[t] = padded[t]
	}
	return out
}

// ComputeChannelsFirstFlat is a convenience wrapper that returns the flat
// `[1, NumMelBins, T_mel]` tensor ready to feed straight to a NeMo ONNX
// session. Returns (flat, tMel). T_mel matches the time dim after pad_to.
func (m *MelSpec) ComputeChannelsFirstFlat(samples []float32) ([]float32, int) {
	tMel, frames := m.computeImpl(samples)
	if tMel == 0 {
		return nil, 0
	}
	flat := make([]float32, m.opts.NumMelBins*tMel)
	for t := 0; t < tMel; t++ {
		row := frames[t]
		for b := 0; b < m.opts.NumMelBins; b++ {
			flat[b*tMel+t] = row[b]
		}
	}
	return flat, tMel
}

// computeImpl runs the full pipeline: preemphasis → reflect-pad → STFT →
// power → mel filterbank → log → pad_to. Returns (T_mel, frames[T_mel][NumMelBins]).
func (m *MelSpec) computeImpl(samples []float32) (int, [][]float32) {
	n := len(samples)
	if n == 0 {
		return 0, nil
	}

	// 1. Pre-emphasize the whole signal once. NeMo's exact formula:
	//    y[0] = x[0]; y[i] = x[i] - preemph * x[i-1] for i >= 1.
	preemphed := make([]float64, n)
	preemphed[0] = float64(samples[0])
	if m.opts.PreemphCoeff != 0 {
		for i := 1; i < n; i++ {
			preemphed[i] = float64(samples[i]) - m.opts.PreemphCoeff*float64(samples[i-1])
		}
	} else {
		for i := 1; i < n; i++ {
			preemphed[i] = float64(samples[i])
		}
	}

	// 2. Reflection pad by NFFT/2 on each side (matches torch.stft center=True
	//    with pad_mode='reflect').
	pad := m.opts.NFFT / 2
	padded := make([]float64, n+2*pad)
	for i := 0; i < pad; i++ {
		// Reflect: padded[i] = preemphed[pad - i] (mirror around index 0).
		padded[i] = preemphed[pad-i]
	}
	copy(padded[pad:pad+n], preemphed)
	for i := 0; i < pad; i++ {
		// Reflect at the right edge: padded[pad+n+i] = preemphed[n-2-i].
		idx := n - 2 - i
		if idx < 0 {
			idx = 0
		}
		padded[pad+n+i] = preemphed[idx]
	}

	// 3. Frame: each frame starts at i*hop in the padded signal.
	//    NeMo's get_seq_len is floor((N + 2*pad - n_fft) / hop) — no +1.
	//    torch.stft natively produces ONE MORE frame than this; NeMo masks
	//    that trailing frame to 0 before pad_to. We match by producing only
	//    NeMo's count up front (omitting the extra trailing frame).
	tMel := (len(padded) - m.opts.NFFT) / m.opts.HopLenSamples
	if tMel <= 0 {
		return 0, nil
	}

	// Compute padded T_mel (after pad_to right-padding with zeros).
	finalT := tMel
	if m.opts.PadToMultiple > 0 {
		rem := finalT % m.opts.PadToMultiple
		if rem != 0 {
			finalT += m.opts.PadToMultiple - rem
		}
	}

	out := make([][]float32, finalT)
	for t := 0; t < tMel; t++ {
		out[t] = make([]float32, m.opts.NumMelBins)
		m.computeFrame(padded[t*m.opts.HopLenSamples:t*m.opts.HopLenSamples+m.opts.NFFT], out[t])
	}
	// pad_to=N right-pads the time axis with LITERAL zeros (matching torch's
	// nn.functional.pad default mode='constant' value=0 — applied to the
	// already-logged feature matrix, NOT log-of-zero). The padded frames
	// look like log-energy = 0 (=> unit linear), which is technically
	// "louder than silence" but the model is trained to handle these padding
	// frames as such.
	for t := tMel; t < finalT; t++ {
		out[t] = make([]float32, m.opts.NumMelBins)
	}
	return finalT, out
}

// computeFrame applies window + FFT + power + mel + log to one frame's worth
// of samples (length must be NFFT). When WinLenSamples < NFFT, the window
// is CENTERED inside the NFFT buffer with (NFFT-WinLenSamples)/2 zeros on
// each side — matching torch.stft's behaviour. Getting this wrong silently
// produces "windows over the wrong 400 samples per frame" which yields a
// numerically-plausible but model-incompatible spectrogram.
func (m *MelSpec) computeFrame(in []float64, out []float32) {
	for i := 0; i < m.opts.NFFT; i++ {
		m.frame[i] = 0
	}
	winOffset := (m.opts.NFFT - m.opts.WinLenSamples) / 2
	for i := 0; i < m.opts.WinLenSamples; i++ {
		m.frame[winOffset+i] = in[winOffset+i] * m.window[i]
	}
	// FFT: real input → complex spectrum.
	for i := 0; i < m.opts.NFFT; i++ {
		m.fftBuf[i] = complex(m.frame[i], 0)
	}
	fftRadix2(m.fftBuf)
	// Power spectrum |X[k]|^2 for k = 0..fftBins.
	power := make([]float64, m.fftBins)
	for k := 0; k < m.fftBins; k++ {
		re := real(m.fftBuf[k])
		im := imag(m.fftBuf[k])
		power[k] = re*re + im*im
	}
	// Slaney mel filterbank * power, then log(mel + log_offset).
	for b := 0; b < m.opts.NumMelBins; b++ {
		row := m.mel[b]
		var sum float64
		for k := 0; k < m.fftBins; k++ {
			sum += row[k] * power[k]
		}
		out[b] = float32(math.Log(sum + m.opts.LogOffset))
	}
}

// NumFrames returns how many output frames Compute will produce for the
// given input length (after pad_to). Useful for sizing buffers.
func (m *MelSpec) NumFrames(numSamples int) int {
	if numSamples <= 0 {
		return 0
	}
	pad := m.opts.NFFT / 2
	t := (numSamples + 2*pad - m.opts.NFFT) / m.opts.HopLenSamples
	if t <= 0 {
		return 0
	}
	if m.opts.PadToMultiple > 0 {
		rem := t % m.opts.PadToMultiple
		if rem != 0 {
			t += m.opts.PadToMultiple - rem
		}
	}
	return t
}

// hannPeriodic returns a Hann window of length L using the "periodic"
// convention: w[i] = 0.5 - 0.5*cos(2π*i/L). torchaudio's hann_window with
// periodic=True (the default).
func hannPeriodic(length int) []float64 {
	w := make([]float64, length)
	if length == 0 {
		return w
	}
	denom := float64(length)
	for i := 0; i < length; i++ {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/denom)
	}
	return w
}
