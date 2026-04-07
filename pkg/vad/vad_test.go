package vad_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/accretional/vad/internal/audio"
	"github.com/accretional/vad/pkg/vad"
)

func ortLibPath() string {
	if p := os.Getenv("ONNXRUNTIME_LIB"); p != "" {
		return p
	}
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")

	var platform string
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			platform = "osx-arm64"
		} else {
			platform = "osx-x86_64"
		}
	case "linux":
		if runtime.GOARCH == "amd64" {
			platform = "linux-x64"
		} else {
			platform = "linux-aarch64"
		}
	}

	libName := "libonnxruntime.so"
	if runtime.GOOS == "darwin" {
		libName = "libonnxruntime.dylib"
	}

	return filepath.Join(repoRoot, "third_party", "onnxruntime-"+platform+"-1.22.0", "lib", libName)
}

func modelPath() string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..", "weights", "model.onnx")
}

func dataPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..", "data", name)
}

func setupModel(t *testing.T) *vad.Model {
	t.Helper()

	lib := ortLibPath()
	if _, err := os.Stat(lib); os.IsNotExist(err) {
		t.Skipf("ONNX Runtime library not found at %s; run setup.sh first", lib)
	}

	mp := modelPath()
	if _, err := os.Stat(mp); os.IsNotExist(err) {
		t.Skipf("Model weights not found at %s; run setup.sh first", mp)
	}

	if err := vad.InitONNXRuntime(lib); err != nil {
		t.Fatalf("InitONNXRuntime: %v", err)
	}

	m, err := vad.NewModel(mp)
	if err != nil {
		vad.DestroyONNXRuntime()
		t.Fatalf("NewModel: %v", err)
	}

	t.Cleanup(func() {
		m.Close()
		vad.DestroyONNXRuntime()
	})

	return m
}

// runAudioTest loads a .f32 file, runs inference, and validates the output.
// minSegments and minSpeakers set minimum expected counts.
func runAudioTest(t *testing.T, m *vad.Model, f32File string, minSegments, minSpeakers int) []vad.Segment {
	t.Helper()

	audioFile := dataPath(f32File)
	if _, err := os.Stat(audioFile); os.IsNotExist(err) {
		t.Skipf("Audio file not found at %s; run encode-to-16k.sh first", audioFile)
	}

	samples, err := audio.LoadF32(audioFile)
	if err != nil {
		t.Fatalf("LoadF32: %v", err)
	}

	if len(samples) < vad.SampleRate {
		t.Fatalf("audio too short: %d samples", len(samples))
	}

	segments, err := m.ProcessAudio(samples)
	if err != nil {
		t.Fatalf("ProcessAudio: %v", err)
	}

	if len(segments) < minSegments {
		t.Errorf("expected at least %d segments, got %d", minSegments, len(segments))
	}

	audioDuration := float64(len(samples)) / float64(vad.SampleRate)
	for i, seg := range segments {
		if seg.Start < 0 || seg.Start >= audioDuration+10 {
			t.Errorf("segment %d: start %.3f out of range", i, seg.Start)
		}
		if seg.End <= seg.Start {
			t.Errorf("segment %d: end %.3f <= start %.3f", i, seg.End, seg.Start)
		}
		if seg.SpeakerID < 0 || seg.SpeakerID > 2 {
			t.Errorf("segment %d: invalid speaker ID %d", i, seg.SpeakerID)
		}
		if seg.Confidence < 0 || seg.Confidence > 1 {
			t.Errorf("segment %d: confidence %.4f out of [0,1]", i, seg.Confidence)
		}
	}

	speakers := make(map[int]bool)
	for _, seg := range segments {
		speakers[seg.SpeakerID] = true
	}
	if len(speakers) < minSpeakers {
		t.Errorf("expected at least %d speakers, got %d", minSpeakers, len(speakers))
	}

	t.Logf("Detected %d segments across %d speakers in %.1fs of audio",
		len(segments), len(speakers), audioDuration)
	for _, seg := range segments {
		t.Logf("  [%.3f - %.3f] speaker_%d (conf: %.4f)",
			seg.Start, seg.End, seg.SpeakerID, seg.Confidence)
	}

	return segments
}

func TestSilence(t *testing.T) {
	m := setupModel(t)

	silence := make([]float32, vad.WindowSize)
	segments, err := m.ProcessAudio(silence)
	if err != nil {
		t.Fatalf("ProcessAudio on silence: %v", err)
	}

	if len(segments) != 0 {
		t.Errorf("expected 0 segments on silence, got %d", len(segments))
	}
}

func TestSorryDave(t *testing.T) {
	m := setupModel(t)
	runAudioTest(t, m, "sorry-dave-16k.f32", 2, 2)
}

func TestBestFriends(t *testing.T) {
	m := setupModel(t)
	runAudioTest(t, m, "bestfriends-16k.f32", 1, 1)
}

func TestWakeMeUp(t *testing.T) {
	m := setupModel(t)
	// This clip is music — the model may detect 0 speech segments, which is valid.
	runAudioTest(t, m, "wake-me-up-16k.f32", 0, 0)
}

func TestLoadF32Audio(t *testing.T) {
	audioFile := dataPath("sorry-dave-16k.f32")
	if _, err := os.Stat(audioFile); os.IsNotExist(err) {
		t.Skipf("Audio file not found; run encode-to-16k.sh first")
	}

	samples, err := audio.LoadF32(audioFile)
	if err != nil {
		t.Fatalf("LoadF32: %v", err)
	}

	expectedMin := 50 * vad.SampleRate
	expectedMax := 60 * vad.SampleRate
	if len(samples) < expectedMin || len(samples) > expectedMax {
		t.Errorf("expected %d-%d samples, got %d", expectedMin, expectedMax, len(samples))
	}

	hasNonZero := false
	for _, s := range samples {
		if s != 0 {
			hasNonZero = true
		}
		if s < -1.5 || s > 1.5 {
			t.Errorf("sample out of range: %f", s)
			break
		}
	}
	if !hasNonZero {
		t.Error("all samples are zero")
	}
}
