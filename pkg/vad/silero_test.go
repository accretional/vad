package vad_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/accretional/vad/pkg/vad"
)

func TestSilero_SmokeRefClip(t *testing.T) {
	weightsDir := repoFile(t, "weights", "silero")
	if _, err := os.Stat(filepath.Join(weightsDir, "model.onnx")); err != nil {
		t.Skipf("silero weights not at %s: %v (run speax/benchmarks/vad/export_silero_to_onnx.py)", weightsDir, err)
	}
	if err := vad.InitONNXRuntime(ortLibPath()); err != nil {
		t.Fatalf("init ort: %v", err)
	}
	t.Cleanup(func() { _ = vad.DestroyONNXRuntime() })

	m, err := vad.NewSilero(weightsDir)
	if err != nil {
		t.Fatalf("NewSilero: %v", err)
	}
	t.Cleanup(m.Close)

	samples := loadF32(t, repoFile(t, "data", "sorry-dave-16k.f32"))
	t.Logf("audio: %.2fs", float64(len(samples))/16000)

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
	t.Logf("silero: %d segments, %.2fs total speech (%.1f%% of clip)",
		len(segs), totalSpeech, 100*frac)
	if frac < 0.2 || frac > 0.95 {
		t.Errorf("speech fraction %.2f outside sanity range [0.2, 0.95]", frac)
	}
	for i := 0; i < len(segs) && i < 5; i++ {
		t.Logf("  seg[%d] %.2f..%.2fs", i, segs[i].Start, segs[i].End)
	}
}
