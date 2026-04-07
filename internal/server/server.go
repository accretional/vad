package server

import (
	"context"
	"encoding/binary"
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
