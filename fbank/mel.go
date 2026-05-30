package fbank

import "math"

// hzToMel converts a frequency in Hz to the HTK Mel scale used by Kaldi.
func hzToMel(hz float64) float64 {
	return 1127.0 * math.Log(1.0+hz/700.0)
}

// melToHz is the inverse of hzToMel.
func melToHz(mel float64) float64 {
	return 700.0 * (math.Exp(mel/1127.0) - 1.0)
}

// makeMelFilterbank builds a [numBins][fftBins] triangular Mel filterbank
// matrix matching Kaldi's `compute-mel-filterbank` defaults.
//
// fftBins is the number of positive-frequency bins (= fftSize/2 + 1).
// The filters span [lowFreq, highFreq] in mel-spaced centers; each filter is
// 0 outside its triangular passband and rises linearly to 1 at its center.
func makeMelFilterbank(numBins, fftBins int, sampleRate, lowFreq, highFreq float64) [][]float64 {
	fftSize := (fftBins - 1) * 2
	if highFreq <= 0 || highFreq > sampleRate/2 {
		highFreq = sampleRate / 2
	}

	// Mel-spaced filter center frequencies (numBins+2 points: one extra at each end).
	melLow := hzToMel(lowFreq)
	melHigh := hzToMel(highFreq)
	melPoints := make([]float64, numBins+2)
	for i := range melPoints {
		melPoints[i] = melLow + (melHigh-melLow)*float64(i)/float64(numBins+1)
	}
	hzPoints := make([]float64, len(melPoints))
	for i, m := range melPoints {
		hzPoints[i] = melToHz(m)
	}

	// FFT bin center frequencies.
	binHz := make([]float64, fftBins)
	for k := 0; k < fftBins; k++ {
		binHz[k] = float64(k) * sampleRate / float64(fftSize)
	}

	filt := make([][]float64, numBins)
	for b := 0; b < numBins; b++ {
		left, center, right := hzPoints[b], hzPoints[b+1], hzPoints[b+2]
		row := make([]float64, fftBins)
		for k := 0; k < fftBins; k++ {
			f := binHz[k]
			if f < left || f > right {
				continue
			}
			if f <= center {
				row[k] = (f - left) / (center - left)
			} else {
				row[k] = (right - f) / (right - center)
			}
		}
		filt[b] = row
	}
	return filt
}
