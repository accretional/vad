package melspec

import "math"

// Slaney piecewise mel scale, matching librosa.hz_to_mel(htk=False).
//
// Linear up to 1000 Hz at f_sp = 200/3 mel/Hz, then logarithmic with
// logstep = ln(6.4)/27.
const (
	slaneyFSp        = 200.0 / 3.0
	slaneyMinLogHz   = 1000.0
	slaneyMinLogMel  = slaneyMinLogHz / slaneyFSp
)

var slaneyLogStep = math.Log(6.4) / 27.0

func hzToMelSlaney(hz float64) float64 {
	if hz < slaneyMinLogHz {
		return hz / slaneyFSp
	}
	return slaneyMinLogMel + math.Log(hz/slaneyMinLogHz)/slaneyLogStep
}

func melToHzSlaney(mel float64) float64 {
	if mel < slaneyMinLogMel {
		return mel * slaneyFSp
	}
	return slaneyMinLogHz * math.Exp(slaneyLogStep*(mel-slaneyMinLogMel))
}

// makeMelFilterbankSlaney builds a `[numBins][fftBins]` triangular Mel
// filterbank with Slaney scale and Slaney normalization (each filter scaled
// by 2 / (right_hz - left_hz) so triangle area is normalized).
//
// Matches `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')`,
// which is what NeMo's MelSpectrogram uses when `mel_norm='slaney'` (the default).
func makeMelFilterbankSlaney(numBins, fftBins int, sampleRate, lowFreq, highFreq float64) [][]float64 {
	fftSize := (fftBins - 1) * 2
	if highFreq <= 0 || highFreq > sampleRate/2 {
		highFreq = sampleRate / 2
	}
	melLow := hzToMelSlaney(lowFreq)
	melHigh := hzToMelSlaney(highFreq)
	melPoints := make([]float64, numBins+2)
	for i := range melPoints {
		melPoints[i] = melLow + (melHigh-melLow)*float64(i)/float64(numBins+1)
	}
	hzPoints := make([]float64, len(melPoints))
	for i, m := range melPoints {
		hzPoints[i] = melToHzSlaney(m)
	}
	binHz := make([]float64, fftBins)
	for k := 0; k < fftBins; k++ {
		binHz[k] = float64(k) * sampleRate / float64(fftSize)
	}
	filt := make([][]float64, numBins)
	for b := 0; b < numBins; b++ {
		left, center, right := hzPoints[b], hzPoints[b+1], hzPoints[b+2]
		// Slaney area normalization: enorm = 2 / (right - left).
		enorm := 2.0 / (right - left)
		row := make([]float64, fftBins)
		for k := 0; k < fftBins; k++ {
			f := binHz[k]
			if f < left || f > right {
				continue
			}
			if f <= center {
				row[k] = (f - left) / (center - left) * enorm
			} else {
				row[k] = (right - f) / (right - center) * enorm
			}
		}
		filt[b] = row
	}
	return filt
}
