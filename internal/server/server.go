package server

import (
	"context"
	"encoding/binary"
	"io"
	"math"

	"github.com/accretional/vad/pkg/vad"
	pb "github.com/accretional/vad/proto/vadpb"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// FetchBytesFunc returns the raw ONNX weight bytes for the given model.
// Supplied at construction; typically backed by internal/embedded.WeightsBytes
// which checks embedded files first and falls back to on-disk weights/.
type FetchBytesFunc func(model pb.VADModel) ([]byte, error)

// FetchURLFunc returns a CDN URL for the given model's .onnx weights, if one
// is available (typically from a url.txt sidecar embedded next to the model).
// Returning ("", false) means no URL is registered for this model and the
// server falls back to streaming bytes via FetchBytesFunc.
type FetchURLFunc func(model pb.VADModel) (string, bool)

// Server implements the VoiceSegmentation gRPC service.
//
// Holds exactly one inference backend (loaded at startup from VADConfig.model).
// Per-request backend selection for Detect/DetectStream is a future change;
// today the Fetch RPC is the only one that accepts a per-request model
// parameter (so clients can pull weights for any embedded/disk backend
// without restarting the server with a different config).
type Server struct {
	pb.UnimplementedVoiceSegmentationServer
	backend      vad.Backend
	defaultModel pb.VADModel
	fetchBytes   FetchBytesFunc
	fetchURL     FetchURLFunc
	weightsURL   string
}

// New creates a new Server.
//
//   - backend: the loaded inference backend (used for Detect / DetectStream).
//   - defaultModel: the enum value matching `backend`; used by Fetch when the
//     client doesn't specify a model.
//   - fetchBytes: closure that returns raw ONNX bytes for ANY backend; called
//     by Fetch when the client wants weights without proxying through the
//     server's CDN-style URL.
//   - fetchURL: optional closure returning a per-model CDN URL when one is
//     registered (e.g. via a url.txt sidecar). When non-nil and the requested
//     model has a URL, Fetch returns that URL and skips streaming bytes.
//   - weightsURL: optional global override. If set AND the Fetch request
//     targets defaultModel AND fetchURL didn't supply a per-model URL, the
//     server redirects clients to this URL instead of streaming bytes.
//     Retained for back-compat with the -weights-url flag.
func New(backend vad.Backend, defaultModel pb.VADModel, fetchBytes FetchBytesFunc, fetchURL FetchURLFunc, weightsURL string) *Server {
	return &Server{
		backend:      backend,
		defaultModel: defaultModel,
		fetchBytes:   fetchBytes,
		fetchURL:     fetchURL,
		weightsURL:   weightsURL,
	}
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

	segments, err := s.backend.ProcessAudio(samples)
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
		segments, err := s.backend.ProcessAudio(buf)
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
//
// Request semantics:
//   - `model` unset / VAD_MODEL_UNSPECIFIED: use the server's defaultModel.
//   - `model` specified: return weights for that backend (works for any
//     backend whose weights are embedded or on disk, not just the one
//     currently loaded for Detect).
//
// Response precedence (first applicable wins):
//  1. Per-model URL from fetchURL (typically a url.txt sidecar) — applies to
//     ANY requested model.
//  2. Global weightsURL configured at startup — applies only when the
//     requested model matches defaultModel (the URL was registered for it).
//  3. Raw bytes from fetchBytes.
func (s *Server) Fetch(ctx context.Context, req *pb.FetchRequest) (*pb.FetchResponse, error) {
	model := req.GetModel()
	if model == pb.VADModel_VAD_MODEL_UNSPECIFIED {
		model = s.defaultModel
	}
	if s.fetchURL != nil {
		if url, ok := s.fetchURL(model); ok {
			return &pb.FetchResponse{
				Result: &pb.FetchResponse_Url{Url: url},
			}, nil
		}
	}
	if model == s.defaultModel && s.weightsURL != "" {
		return &pb.FetchResponse{
			Result: &pb.FetchResponse_Url{Url: s.weightsURL},
		}, nil
	}
	if s.fetchBytes == nil {
		return nil, status.Error(codes.Unimplemented, "server has no weights fetcher configured")
	}
	data, err := s.fetchBytes(model)
	if err != nil {
		return nil, status.Errorf(codes.NotFound, "weights for %s: %v", model.String(), err)
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
