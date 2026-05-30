package fbank

import "math"

// WindowType selects the windowing function applied before FFT.
type WindowType string

const (
	// WindowHamming = 0.54 - 0.46*cos(2π n / (N-1)). FunASR / FSMN front-end uses this.
	WindowHamming WindowType = "hamming"
	// WindowPovey = (0.5 - 0.5*cos(2π n / (N-1)))^0.85. Kaldi default; FireRedVAD uses this.
	WindowPovey WindowType = "povey"
)

// makeWindow precomputes a window of the given length and type. Output is
// reused per frame.
func makeWindow(length int, t WindowType) []float64 {
	w := make([]float64, length)
	if length == 1 {
		w[0] = 1
		return w
	}
	denom := float64(length - 1)
	switch t {
	case WindowHamming:
		for i := 0; i < length; i++ {
			w[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/denom)
		}
	case WindowPovey:
		for i := 0; i < length; i++ {
			h := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/denom)
			w[i] = math.Pow(h, 0.85)
		}
	default:
		// Default to hamming if an unknown type is passed.
		for i := 0; i < length; i++ {
			w[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/denom)
		}
	}
	return w
}
