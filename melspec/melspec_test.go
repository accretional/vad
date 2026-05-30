package melspec_test

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/accretional/vad/melspec"
)

const (
	numMels      = 80
	parityTolP99 = 1e-1  // log-mel values span ~[-16, 8]; 0.1 in log space ≈ 10% in linear
	parityTolMax = 5e-1
)

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

func loadFixture(t *testing.T) [][]float32 {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(thisFile), "testdata", "nemo-sorry-dave.f32")
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	st, _ := f.Stat()
	if st.Size()%(4*numMels) != 0 {
		t.Fatalf("fixture size %d not divisible by 4*%d", st.Size(), numMels)
	}
	frames := int(st.Size()) / (4 * numMels)
	out := make([][]float32, frames)
	buf := make([]byte, 4*numMels)
	for i := 0; i < frames; i++ {
		if _, err := io.ReadFull(f, buf); err != nil {
			t.Fatalf("read frame %d: %v", i, err)
		}
		row := make([]float32, numMels)
		for j := 0; j < numMels; j++ {
			row[j] = math.Float32frombits(binary.LittleEndian.Uint32(buf[j*4:]))
		}
		out[i] = row
	}
	return out
}

func TestParity_NeMo(t *testing.T) {
	audio := loadAudioF32(t)
	want := loadFixture(t)

	m, err := melspec.New(melspec.NeMoDefaults())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := m.Compute(audio)

	if len(got) != len(want) {
		t.Fatalf("frame count: got %d, want %d", len(got), len(want))
	}
	if len(got) == 0 {
		t.Fatalf("got 0 frames")
	}

	var diffs []float64
	var maxAbs float64
	for i := range got {
		for j := range got[i] {
			d := math.Abs(float64(got[i][j] - want[i][j]))
			diffs = append(diffs, d)
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	sortFloats(diffs)
	p99 := diffs[int(float64(len(diffs)-1)*0.99)]
	mean := func() float64 {
		var s float64
		for _, d := range diffs {
			s += d
		}
		return s / float64(len(diffs))
	}()
	t.Logf("parity: frames=%d mels=%d max_abs=%.4e p99=%.4e mean=%.4e",
		len(got), numMels, maxAbs, p99, mean)
	if p99 > parityTolP99 {
		t.Errorf("p99 abs diff %.4e exceeds %.4e", p99, parityTolP99)
	}
	if maxAbs > parityTolMax {
		t.Errorf("max abs diff %.4e exceeds %.4e", maxAbs, parityTolMax)
	}

	// Spot-check a few "non-silence" frames to make sure we're not just
	// matching floor everywhere.
	for _, fi := range []int{100, 500, 1000, 3000} {
		if fi >= len(got) {
			continue
		}
		var maxFrame float64
		for j := 0; j < numMels; j++ {
			d := math.Abs(float64(got[fi][j] - want[fi][j]))
			if d > maxFrame {
				maxFrame = d
			}
		}
		t.Logf("  frame %d: max_abs=%.4e  got[:3]=%v  want[:3]=%v",
			fi, maxFrame, got[fi][:3], want[fi][:3])
	}
}

func sortFloats(a []float64) {
	// Tiny insertion sort; only used in tests.
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
