package audio

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// SaveF32 writes float32 samples to a raw f32le file.
func SaveF32(path string, samples []float32) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	buf := make([]byte, 4)
	for _, s := range samples {
		binary.LittleEndian.PutUint32(buf, math.Float32bits(s))
		if _, err := f.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

// SaveSegmented writes each speaker chunk to basePath-speaker{N}-segment{M}.f32
func SaveSegmented(basePath string, chunks []SpeakerChunk) error {
	for _, chunk := range chunks {
		path := fmt.Sprintf("%s-speaker%d-segment%d.f32",
			basePath, chunk.SpeakerID, chunk.SegmentIdx)
		if err := SaveF32(path, chunk.Samples); err != nil {
			return fmt.Errorf("save chunk speaker%d segment%d: %w",
				chunk.SpeakerID, chunk.SegmentIdx, err)
		}
	}
	return nil
}

// SaveUnsegmented writes each speaker stream to basePath-speaker{N}.f32
func SaveUnsegmented(basePath string, streams []SpeakerStream) error {
	for _, stream := range streams {
		path := fmt.Sprintf("%s-speaker%d.f32", basePath, stream.SpeakerID)
		if err := SaveF32(path, stream.Samples); err != nil {
			return fmt.Errorf("save stream speaker%d: %w", stream.SpeakerID, err)
		}
	}
	return nil
}
