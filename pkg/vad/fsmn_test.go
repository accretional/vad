package vad_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/accretional/vad/pkg/vad"
)

func TestFSMN_SmokeRefClip(t *testing.T) {
	weightsDir := repoFile(t, "weights", "fsmn-vad")
	if _, err := os.Stat(filepath.Join(weightsDir, "model.onnx")); err != nil {
		t.Skipf("fsmn-vad weights not at %s: %v (run setup.sh)", weightsDir, err)
	}
	if err := vad.InitONNXRuntime(ortLibPath()); err != nil {
		t.Fatalf("init ort: %v", err)
	}
	t.Cleanup(func() { _ = vad.DestroyONNXRuntime() })

	m, err := vad.NewFSMN(weightsDir)
	if err != nil {
		t.Fatalf("NewFSMN: %v", err)
	}
	t.Cleanup(m.Close)

	samples := loadF32(t, repoFile(t, "data", "sorry-dave-16k.f32"))
	t.Logf("audio: %.2f s, %d samples", float64(len(samples))/16000, len(samples))

	segs, err := m.ProcessAudio(samples)
	if err != nil {
		t.Fatalf("ProcessAudio: %v", err)
	}
	if len(segs) == 0 {
		t.Fatalf("got 0 segments; expected speech to be detected")
	}
	var totalSpeech float64
	for _, s := range segs {
		if s.End <= s.Start {
			t.Errorf("bad segment %v", s)
		}
		totalSpeech += s.End - s.Start
	}
	audioDur := float64(len(samples)) / 16000
	frac := totalSpeech / audioDur
	t.Logf("fsmn: %d segments, %.2fs total speech (%.1f%% of clip)",
		len(segs), totalSpeech, 100*frac)
	if frac < 0.2 || frac > 0.95 {
		t.Errorf("speech fraction %.2f outside sanity range [0.2, 0.95]", frac)
	}
	for i := 0; i < len(segs) && i < 5; i++ {
		t.Logf("  seg[%d] %.2f..%.2fs speaker=%d", i, segs[i].Start, segs[i].End, segs[i].SpeakerID)
	}
}

// repoFile resolves a path relative to the vad repo root.
func repoFile(t *testing.T, parts ...string) string {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")
	return filepath.Join(append([]string{repoRoot}, parts...)...)
}

// loadF32 reads raw 16 kHz mono float32 audio.
func loadF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read audio: %v", err)
	}
	n := len(raw) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}
