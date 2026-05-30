package server

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"os"

	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Server implements the VoiceSegmentation gRPC service.
type Server struct {
	pb.UnimplementedVoiceSegmentationServer
	model      *vad.Model
	weightsURL string
	modelPath  string
}

// New creates a new Server with the given VAD model.
// weightsURL is optional — if set, Fetch returns it instead of the raw weights.
// modelPath is needed to read weights from disk when weightsURL is empty.
func New(model *vad.Model, modelPath string, weightsURL string) *Server {
	return &Server{model: model, modelPath: modelPath, weightsURL: weightsURL}
}

// Detect processes raw audio and returns speaker-diarized segments.
func (s *Server) Detect(ctx context.Context, req *pb.Audio) (*pb.Diarization, error) {
	if req.SampleRate != 0 && req.SampleRate != vad.SampleRate {
		return nil, status.Errorf(codes.InvalidArgument,
			"sample rate must be %d, got %d", vad.SampleRate, req.SampleRate)
	}

	if len(req.Samples) == 0 {
		return nil, status.Error(codes.InvalidArgument, "samples cannot be empty")
	}

	if len(req.Samples)%4 != 0 {
		return nil, status.Errorf(codes.InvalidArgument,
			"samples length %d is not a multiple of 4 (expected float32 little-endian)", len(req.Samples))
	}

	samples := bytesToFloat32(req.Samples)
	duration := float64(len(samples)) / float64(vad.SampleRate)

	segments, err := s.model.ProcessAudio(samples)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "inference failed: %v", err)
	}

	pbSegments := make([]*pb.Segment, len(segments))
	for i, seg := range segments {
		pbSegments[i] = &pb.Segment{
			Start:      seg.Start,
			End:        seg.End,
			SpeakerId:  int32(seg.SpeakerID),
			Confidence: seg.Confidence,
		}
	}

	return &pb.Diarization{
		Segments: pbSegments,
		Duration: duration,
	}, nil
}

// streamConfig controls the rolling-buffer analysis used by DetectStream when
// the backend doesn't expose native streaming. Tunables are exposed as package
// vars (not flags) so callers in tests can poke at them.
var (
	// streamMaxBufferSec caps the rolling buffer; older audio is dropped.
	streamMaxBufferSec float64 = 30.0
	// streamMinAnalyzeSec is the minimum amount of NEW audio that must accumulate
	// before we rerun inference. Smaller = lower endpointing latency, higher CPU.
	streamMinAnalyzeSec float64 = 0.4
	// streamSegmentFinalizeSec is how long a segment's end must be in the past
	// before we emit it as a completed Segment event (gives the rolling analysis
	// a chance to extend the segment if speech continues).
	streamSegmentFinalizeSec float64 = 0.5
)

// DetectStream is a bidi RPC: client sends AudioChunks as audio is captured;
// server emits SegmentationEvents (activity transitions + completed segments)
// in real time. This scaffolding works on any backend that exposes a unary
// ProcessAudio — it accumulates a rolling buffer and re-runs inference. Per
// backend native streaming (e.g. FSMN's chunk_size, FireRed Stream-VAD) is a
// later optimization that would replace this loop with an incremental call.
func (s *Server) DetectStream(stream pb.VoiceSegmentation_DetectStreamServer) error {
	const sampleRate = vad.SampleRate
	maxSamples := int(streamMaxBufferSec * float64(sampleRate))
	minAnalyzeSamples := int(streamMinAnalyzeSec * float64(sampleRate))

	var (
		buf                      []float32
		bufStartTimeS            float64
		speechActive             bool
		samplesSinceLastAnalysis int
		// Track which segments have been emitted (keyed by absolute start time).
		emittedStarts = map[float64]bool{}
	)

	flushAndCheck := func(endOfStream bool) error {
		if len(buf) == 0 {
			return nil
		}
		segments, err := s.model.ProcessAudio(buf)
		if err != nil {
			return status.Errorf(codes.Internal, "inference failed: %v", err)
		}

		bufDurS := float64(len(buf)) / float64(sampleRate)
		nowAbsS := bufStartTimeS + bufDurS

		currentlyActive := false
		for _, seg := range segments {
			absStart := bufStartTimeS + seg.Start
			absEnd := bufStartTimeS + seg.End

			// "Currently speaking" if the most recent segment reaches the
			// end of the analyzed buffer (within 100ms tolerance).
			if absEnd >= nowAbsS-0.1 {
				currentlyActive = true
			}

			// Emit a completed Segment event once its end is safely in the
			// past (segment is unlikely to be extended by future analysis).
			finalize := endOfStream || (nowAbsS-absEnd) >= streamSegmentFinalizeSec
			if finalize && !emittedStarts[absStart] {
				emittedStarts[absStart] = true
				if err := stream.Send(&pb.SegmentationEvent{
					Timestamp: absEnd,
					Event: &pb.SegmentationEvent_Segment{
						Segment: &pb.Segment{
							Start:      absStart,
							End:        absEnd,
							SpeakerId:  int32(seg.SpeakerID),
							Confidence: seg.Confidence,
						},
					},
				}); err != nil {
					return err
				}
			}
		}

		// Emit activity transitions.
		if currentlyActive != speechActive {
			speechActive = currentlyActive
			if err := stream.Send(&pb.SegmentationEvent{
				Timestamp: nowAbsS,
				Event: &pb.SegmentationEvent_Activity{
					Activity: &pb.SpeechActivity{SpeechActive: speechActive},
				},
			}); err != nil {
				return err
			}
		}

		// On end-of-stream, force a final inactive transition if needed.
		if endOfStream && speechActive {
			speechActive = false
			if err := stream.Send(&pb.SegmentationEvent{
				Timestamp: nowAbsS,
				Event: &pb.SegmentationEvent_Activity{
					Activity: &pb.SpeechActivity{SpeechActive: false},
				},
			}); err != nil {
				return err
			}
		}
		return nil
	}

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if chunk.SampleRate != 0 && chunk.SampleRate != sampleRate {
			return status.Errorf(codes.InvalidArgument,
				"sample rate must be %d, got %d", sampleRate, chunk.SampleRate)
		}
		if len(chunk.Samples)%4 != 0 {
			return status.Errorf(codes.InvalidArgument,
				"samples length %d is not a multiple of 4 (expected float32 little-endian)",
				len(chunk.Samples))
		}

		newSamples := bytesToFloat32(chunk.Samples)
		buf = append(buf, newSamples...)
		samplesSinceLastAnalysis += len(newSamples)

		// Slide window: drop oldest samples if the buffer grew past the cap.
		if len(buf) > maxSamples {
			trim := len(buf) - maxSamples
			buf = buf[trim:]
			bufStartTimeS += float64(trim) / float64(sampleRate)
		}

		shouldAnalyze := chunk.EndOfStream || samplesSinceLastAnalysis >= minAnalyzeSamples
		if !shouldAnalyze {
			continue
		}
		samplesSinceLastAnalysis = 0

		if err := flushAndCheck(chunk.EndOfStream); err != nil {
			return err
		}
		if chunk.EndOfStream {
			break
		}
	}
	return nil
}

// Fetch returns the ONNX model weights or a URL to download them.
func (s *Server) Fetch(ctx context.Context, req *pb.FetchRequest) (*pb.FetchResponse, error) {
	if s.weightsURL != "" {
		return &pb.FetchResponse{
			Result: &pb.FetchResponse_Url{Url: s.weightsURL},
		}, nil
	}

	data, err := os.ReadFile(s.modelPath)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to read model weights: %v", err)
	}

	return &pb.FetchResponse{
		Result: &pb.FetchResponse_Weights{Weights: data},
	}, nil
}

func bytesToFloat32(data []byte) []float32 {
	n := len(data) / 4
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		samples[i] = math.Float32frombits(bits)
	}
	return samples
}
