package audio

import (
	"github.com/accretional/vad/pkg/vad"
)

// SpeakerChunk is a single contiguous audio segment for a speaker.
type SpeakerChunk struct {
	SpeakerID  int
	SegmentIdx int
	Start      float64 // seconds
	End        float64 // seconds
	Samples    []float32
}

// SpeakerStream is a full-length audio stream for a single speaker,
// with silence where the speaker is not active.
type SpeakerStream struct {
	SpeakerID int
	Samples   []float32
}

// Segmented applies VAD segments to split audio into per-speaker chunks.
// Each chunk corresponds to one segment from the model output.
func Segmented(samples []float32, segments []vad.Segment) []SpeakerChunk {
	var chunks []SpeakerChunk

	// Track segment index per speaker
	speakerSegIdx := make(map[int]int)

	for _, seg := range segments {
		startSample := int(seg.Start * float64(vad.SampleRate))
		endSample := int(seg.End * float64(vad.SampleRate))

		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(samples) {
			endSample = len(samples)
		}
		if startSample >= endSample {
			continue
		}

		idx := speakerSegIdx[seg.SpeakerID]
		speakerSegIdx[seg.SpeakerID] = idx + 1

		chunk := make([]float32, endSample-startSample)
		copy(chunk, samples[startSample:endSample])

		chunks = append(chunks, SpeakerChunk{
			SpeakerID:  seg.SpeakerID,
			SegmentIdx: idx,
			Start:      seg.Start,
			End:        seg.End,
			Samples:    chunk,
		})
	}

	return chunks
}

// Unsegmented creates full-length audio streams for each speaker.
// Each stream is the same length as the original, with silence where
// the speaker is not active.
func Unsegmented(samples []float32, segments []vad.Segment) []SpeakerStream {
	// Find unique speakers
	speakerSet := make(map[int]bool)
	for _, seg := range segments {
		speakerSet[seg.SpeakerID] = true
	}

	var streams []SpeakerStream
	for speakerID := range speakerSet {
		stream := SpeakerStream{
			SpeakerID: speakerID,
			Samples:   make([]float32, len(samples)),
		}

		for _, seg := range segments {
			if seg.SpeakerID != speakerID {
				continue
			}

			startSample := int(seg.Start * float64(vad.SampleRate))
			endSample := int(seg.End * float64(vad.SampleRate))

			if startSample < 0 {
				startSample = 0
			}
			if endSample > len(samples) {
				endSample = len(samples)
			}

			copy(stream.Samples[startSample:endSample], samples[startSample:endSample])
		}

		streams = append(streams, stream)
	}

	return streams
}
