package fbank_test

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/accretional/vad/fbank"
)

const (
	numMels      = 80
	// Tolerances reflect what's achievable when computing with float64 FFT
	// against kaldi-native-fbank's float32 implementation. Differences are
	// in the last bit or two of float32 — well below model-input precision.
	parityTolP99 = 5e-2
	parityTolMax = 1e-1
)

// loadAudioF32 reads the bundled raw 16 kHz mono float32 audio.
func loadAudioF32(t *testing.T) []float32 {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(thisFile), "..", "data", "sorry-dave-16k.f32")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read audio: %v", err)
	}
	n := len(data) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

// loadFixture reads a [T, numMels] float32 reference fixture written by
// testdata/gen_fixtures.py.
func loadFixture(t *testing.T, label string) [][]float32 {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(thisFile), "testdata", label+".f32")
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	st, _ := f.Stat()
	n := int(st.Size()) / 4
	if n%numMels != 0 {
		t.Fatalf("fixture %s has %d floats, not a multiple of %d", label, n, numMels)
	}
	frames := n / numMels
	out := make([][]float32, frames)
	buf := make([]byte, numMels*4)
	for i := 0; i < frames; i++ {
		if _, err := io.ReadFull(f, buf); err != nil {
			t.Fatalf("read fixture frame %d: %v", i, err)
		}
		row := make([]float32, numMels)
		for j := 0; j < numMels; j++ {
			row[j] = math.Float32frombits(binary.LittleEndian.Uint32(buf[j*4:]))
		}
		out[i] = row
	}
	return out
}

// compareFrames reports parity stats between Go output and the Kaldi fixture.
// Fails the test if either threshold is exceeded.
func compareFrames(t *testing.T, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("frame count: got %d, want %d", len(got), len(want))
	}
	var diffs []float64
	var maxAbs float64
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("frame %d mel count: got %d want %d", i, len(got[i]), len(want[i]))
		}
		for j := range got[i] {
			d := math.Abs(float64(got[i][j] - want[i][j]))
			diffs = append(diffs, d)
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	// p99
	sorted := append([]float64(nil), diffs...)
	sortFloats(sorted)
	p99 := sorted[int(float64(len(sorted)-1)*0.99)]
	t.Logf("parity: frames=%d mels=%d max_abs=%.3e p99=%.3e", len(got), numMels, maxAbs, p99)
	if p99 > parityTolP99 {
		t.Errorf("p99 abs diff %.3e exceeds %.3e", p99, parityTolP99)
	}
	if maxAbs > parityTolMax {
		t.Errorf("max abs diff %.3e exceeds %.3e", maxAbs, parityTolMax)
	}
}

func sortFloats(a []float64) {
	for i := 1; i < len(a); i++ {
		k := a[i]
		j := i - 1
		for j >= 0 && a[j] > k {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = k
	}
}

// runParity runs the fbank with the given options against a labeled fixture.
func runParity(t *testing.T, label string, opts fbank.Options, scale float32) {
	audio := loadAudioF32(t)
	scaled := make([]float32, len(audio))
	for i := range audio {
		scaled[i] = audio[i] * scale
	}
	fb, err := fbank.New(opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := fb.Compute(scaled)
	want := loadFixture(t, label)
	compareFrames(t, got, want)
}

func TestParity_Povey(t *testing.T) {
	opts := fbank.Defaults()
	opts.WindowType = fbank.WindowPovey
	runParity(t, "povey-25-10-80-pre097-scale32768", opts, 32768)
}

func TestParity_Hamming(t *testing.T) {
	opts := fbank.Defaults()
	opts.WindowType = fbank.WindowHamming
	runParity(t, "hamming-25-10-80-pre097-scale32768", opts, 32768)
}

// BenchmarkCompute_10s measures wall time to extract fbank features for 10s of
// audio. The reference kaldi-native-fbank Python impl runs the same workload
// in ~5-10 ms on M-series; see fbank/README.md for the comparison.
func BenchmarkCompute_10s(b *testing.B) {
	_, thisFile, _, _ := runtime.Caller(0)
	audioPath := filepath.Join(filepath.Dir(thisFile), "..", "data", "sorry-dave-16k.f32")
	raw, err := os.ReadFile(audioPath)
	if err != nil {
		b.Fatalf("read audio: %v", err)
	}
	full := make([]float32, len(raw)/4)
	for i := range full {
		full[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])) * 32768
	}
	// Trim to ~10s.
	if len(full) > 10*16000 {
		full = full[:10*16000]
	}
	opts := fbank.Defaults()
	fb, _ := fbank.New(opts)
	// warm
	_ = fb.Compute(full)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fb.Compute(full)
	}
	b.ReportMetric(float64(b.N)*10/(float64(b.Elapsed())/float64(time.Second)), "rt_x")
}
