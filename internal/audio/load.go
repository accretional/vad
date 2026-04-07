package audio

import (
	"encoding/binary"
	"io"
	"math"
	"os"
)

// LoadF32 reads a raw float32 little-endian PCM file (as produced by
// ffmpeg -f f32le -ar 16000 -ac 1).
func LoadF32(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	n := len(data) / 4
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		samples[i] = math.Float32frombits(bits)
	}
	return samples, nil
}
