// Package fbank computes log Mel-filterbank features from 16 kHz mono audio,
// matching Kaldi's `compute-fbank-feats` defaults closely enough for VAD/ASR
// inputs. Parity against `kaldi_native_fbank` is verified in fbank_test.go.
//
// Used by the FSMN-VAD and FireRedVAD backends in pkg/vad. Speech models built
// on top of Kaldi-style features (which is most of them, including FunASR's
// FSMN-VAD and FireRedTeam's DFSMN-VAD) expect specific window shape, FFT
// size, Mel scale, and log floor; getting any of these wrong produces
// numerically-plausible-looking features that the model interprets as noise.
package fbank

import (
	"fmt"
	"math"
)

// Options configure a Fbank instance. Defaults() returns Kaldi's standard
// VAD/ASR settings; override individual fields as needed.
type Options struct {
	SampleRate     int
	FrameLengthMs  float64 // typical 25
	FrameShiftMs   float64 // typical 10
	NumMelBins     int     // typical 80
	WindowType     WindowType
	LowFreqHz      float64 // typical 20
	HighFreqHz     float64 // 0 means SampleRate/2
	PreemphCoeff   float64 // typical 0.97; set to 0 to disable
	RemoveDCOffset bool    // typical true
	SnipEdges      bool    // typical true; false = reflect-pad at edges
	EnergyFloor    float64 // typical 0; the natural-log floor applied to mel energies (clamped to >= this)
	UseLog         bool    // typical true; if false, return raw mel energies (rarely used)
}

// Defaults returns Kaldi compute-fbank-feats default options for 16 kHz audio
// with 80 mel bins and Povey window.
func Defaults() Options {
	return Options{
		SampleRate:     16000,
		FrameLengthMs:  25,
		FrameShiftMs:   10,
		NumMelBins:     80,
		WindowType:     WindowPovey,
		LowFreqHz:      20,
		HighFreqHz:     0, // => SampleRate/2
		PreemphCoeff:   0.97,
		RemoveDCOffset: true,
		SnipEdges:      true,
		EnergyFloor:    0,
		UseLog:         true,
	}
}

// Fbank computes Kaldi-compatible log-mel-filterbank features. Construct once
// (window + filterbank are precomputed); Compute is safe to call repeatedly.
// Not safe for concurrent calls on the same instance — clone if needed.
type Fbank struct {
	opts        Options
	frameLength int
	frameShift  int
	fftSize     int
	fftBins     int // fftSize/2 + 1
	window      []float64
	mel         [][]float64 // [NumMelBins][fftBins]

	// Scratch buffers reused across frames.
	frameBuf []float64
	fftBuf   []complex128
}

// New constructs a Fbank with the given options. Returns an error for
// degenerate options (e.g. zero frame length).
func New(opts Options) (*Fbank, error) {
	if opts.SampleRate <= 0 {
		return nil, fmt.Errorf("fbank: SampleRate must be > 0")
	}
	if opts.FrameLengthMs <= 0 || opts.FrameShiftMs <= 0 {
		return nil, fmt.Errorf("fbank: FrameLengthMs and FrameShiftMs must be > 0")
	}
	if opts.NumMelBins <= 0 {
		return nil, fmt.Errorf("fbank: NumMelBins must be > 0")
	}
	frameLength := int(math.Round(float64(opts.SampleRate) * opts.FrameLengthMs / 1000))
	frameShift := int(math.Round(float64(opts.SampleRate) * opts.FrameShiftMs / 1000))
	fftSize := nextPow2(frameLength)
	fftBins := fftSize/2 + 1

	high := opts.HighFreqHz
	if high <= 0 {
		high = float64(opts.SampleRate) / 2
	}

	return &Fbank{
		opts:        opts,
		frameLength: frameLength,
		frameShift:  frameShift,
		fftSize:     fftSize,
		fftBins:     fftBins,
		window:      makeWindow(frameLength, opts.WindowType),
		mel:         makeMelFilterbank(opts.NumMelBins, fftBins, float64(opts.SampleRate), opts.LowFreqHz, high),
		frameBuf:    make([]float64, fftSize),
		fftBuf:      make([]complex128, fftSize),
	}, nil
}

// NumFrames returns how many output frames Compute will produce for a sample
// vector of the given length.
func (f *Fbank) NumFrames(numSamples int) int {
	if f.opts.SnipEdges {
		if numSamples < f.frameLength {
			return 0
		}
		return (numSamples-f.frameLength)/f.frameShift + 1
	}
	// snip_edges=false: Kaldi uses center-frame alignment with reflection padding.
	// Not used by our current backends; not implemented here.
	return (numSamples + f.frameShift/2) / f.frameShift
}

// Compute returns [NumFrames(len(samples))][NumMelBins] log-mel features.
// The input is mono PCM, float32 valued in the same scale as Kaldi expects
// (e.g. raw int16 values cast to float, or float32 PCM multiplied by 32768
// to match — FunASR and FireRed front-ends both scale [-1,1] floats up).
func (f *Fbank) Compute(samples []float32) [][]float32 {
	nFrames := f.NumFrames(len(samples))
	if nFrames == 0 {
		return nil
	}
	out := make([][]float32, nFrames)
	for i := range out {
		out[i] = make([]float32, f.opts.NumMelBins)
	}

	for fi := 0; fi < nFrames; fi++ {
		start := fi * f.frameShift
		f.computeFrame(samples[start:start+f.frameLength], out[fi])
	}
	return out
}

func (f *Fbank) computeFrame(in []float32, out []float32) {
	// Copy + remove DC + preemphasize, then window.
	for i := 0; i < f.frameLength; i++ {
		f.frameBuf[i] = float64(in[i])
	}
	for i := f.frameLength; i < f.fftSize; i++ {
		f.frameBuf[i] = 0
	}

	if f.opts.RemoveDCOffset {
		var mean float64
		for i := 0; i < f.frameLength; i++ {
			mean += f.frameBuf[i]
		}
		mean /= float64(f.frameLength)
		for i := 0; i < f.frameLength; i++ {
			f.frameBuf[i] -= mean
		}
	}

	if f.opts.PreemphCoeff > 0 {
		// Kaldi pre-emphasis: y[0] = x[0] - coeff*x[0]; y[i] = x[i] - coeff*x[i-1]
		coeff := f.opts.PreemphCoeff
		prev := f.frameBuf[0]
		f.frameBuf[0] = prev - coeff*prev
		for i := 1; i < f.frameLength; i++ {
			cur := f.frameBuf[i]
			f.frameBuf[i] = cur - coeff*prev
			prev = cur
		}
	}

	for i := 0; i < f.frameLength; i++ {
		f.frameBuf[i] *= f.window[i]
	}

	// Real → complex FFT input.
	for i := 0; i < f.fftSize; i++ {
		f.fftBuf[i] = complex(f.frameBuf[i], 0)
	}
	fftRadix2(f.fftBuf)

	// Power spectrum: |X[k]|^2 for the first fftBins (positive-freq half).
	power := make([]float64, f.fftBins) // small alloc; we could pool if hot
	for k := 0; k < f.fftBins; k++ {
		re := real(f.fftBuf[k])
		im := imag(f.fftBuf[k])
		power[k] = re*re + im*im
	}

	// Mel filterbank * power, then log. Match kaldi-native-fbank, which uses
	// std::numeric_limits<float>::epsilon() (≈ 1.1920929e-7) as the implicit
	// floor before log — so log of "silent" frames is -15.942 (not -23 or
	// whatever a smaller floor would produce). Tested via parity fixture.
	const float32Epsilon = 1.1920928955078125e-7
	floor := f.opts.EnergyFloor
	if floor < float32Epsilon {
		floor = float32Epsilon
	}
	for b := 0; b < f.opts.NumMelBins; b++ {
		row := f.mel[b]
		var sum float64
		for k := 0; k < f.fftBins; k++ {
			sum += row[k] * power[k]
		}
		if f.opts.UseLog {
			if sum < floor {
				sum = floor
			}
			out[b] = float32(math.Log(sum))
		} else {
			out[b] = float32(sum)
		}
	}
}
