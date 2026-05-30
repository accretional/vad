// Package vad provides voice activity detection backends. Multiple model
// implementations (Pyannote Segmentation 3.0, FSMN-VAD, FireRedVAD) plug into
// a common Backend interface; the gRPC server selects one at startup via the
// -backend flag (see cmd/vad).
package vad

import (
	ort "github.com/yalue/onnxruntime_go"
)

// Segment represents a detected voice activity region. Speaker-aware backends
// (currently only Pyannote) populate SpeakerID; VAD-only backends emit
// SpeakerID=0 for every segment.
type Segment struct {
	Start      float64 // seconds from start of input audio
	End        float64 // seconds from start of input audio
	SpeakerID  int     // 0-indexed; 0 for backends without speaker labels
	Confidence float32 // 0.0 - 1.0; 1.0 for backends without per-segment confidence
}

// SampleRate is the expected input audio sample rate for every backend.
const SampleRate = 16000

// Backend is the common interface implemented by every VAD model.
//
// ProcessAudio runs the backend on the given mono 16 kHz float32 PCM samples
// and returns the detected speech segments. Implementations are expected to be
// goroutine-safe via internal locking.
//
// Close releases any held resources (ONNX session, tensors, etc.).
type Backend interface {
	ProcessAudio(samples []float32) ([]Segment, error)
	Close()
}

// InitONNXRuntime initializes ONNX Runtime with the given shared library
// (libonnxruntime.dylib on macOS, libonnxruntime.so on linux). Must be called
// once before any backend constructor.
func InitONNXRuntime(libPath string) error {
	ort.SetSharedLibraryPath(libPath)
	return ort.InitializeEnvironment()
}

// DestroyONNXRuntime cleans up the ONNX Runtime environment.
func DestroyONNXRuntime() error {
	return ort.DestroyEnvironment()
}
