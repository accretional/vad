package server

import (
	"context"
	"errors"
	"testing"

	pb "github.com/accretional/vad/proto/vadpb"
)

// TestFetchPrefersURLOverBytes confirms the URL-precedence wiring: when
// fetchURL returns (url, true) for the requested model, Fetch returns that
// URL and skips fetchBytes entirely. This is the path the browser demo uses
// to redirect clients to a CDN download instead of streaming weights through
// the gRPC pipe.
func TestFetchPrefersURLOverBytes(t *testing.T) {
	const wantURL = "https://example.com/weights/pyannote.onnx"
	bytesCalled := false
	s := New(
		nil, // backend not needed for Fetch
		pb.VADModel_VAD_MODEL_PYANNOTE,
		func(m pb.VADModel) ([]byte, error) {
			bytesCalled = true
			return []byte("should-not-be-returned"), nil
		},
		func(m pb.VADModel) (string, bool) {
			if m == pb.VADModel_VAD_MODEL_PYANNOTE {
				return wantURL, true
			}
			return "", false
		},
		"",
	)
	resp, err := s.Fetch(context.Background(), &pb.FetchRequest{
		Model: pb.VADModel_VAD_MODEL_PYANNOTE,
	})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if got := resp.GetUrl(); got != wantURL {
		t.Errorf("got url %q, want %q", got, wantURL)
	}
	if len(resp.GetWeights()) != 0 {
		t.Errorf("got %d weight bytes, want 0 (url branch should leave weights empty)", len(resp.GetWeights()))
	}
	if bytesCalled {
		t.Error("fetchBytes was called even though fetchURL returned a URL")
	}
}

// TestFetchFallsBackToBytes confirms that when fetchURL returns (_, false)
// for the requested model, Fetch falls through to fetchBytes and streams the
// raw ONNX payload. This is the path for backends without a url.txt sidecar.
func TestFetchFallsBackToBytes(t *testing.T) {
	wantBytes := []byte{0x01, 0x02, 0x03, 0x04}
	s := New(
		nil,
		pb.VADModel_VAD_MODEL_FSMN,
		func(m pb.VADModel) ([]byte, error) {
			if m != pb.VADModel_VAD_MODEL_FSMN {
				return nil, errors.New("unexpected model")
			}
			return wantBytes, nil
		},
		func(m pb.VADModel) (string, bool) {
			return "", false // no url.txt for any model
		},
		"",
	)
	resp, err := s.Fetch(context.Background(), &pb.FetchRequest{
		Model: pb.VADModel_VAD_MODEL_FSMN,
	})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if resp.GetUrl() != "" {
		t.Errorf("got url %q, want empty (bytes branch should leave url empty)", resp.GetUrl())
	}
	if got := resp.GetWeights(); string(got) != string(wantBytes) {
		t.Errorf("got weights %v, want %v", got, wantBytes)
	}
}

// TestFetchHonoursLegacyWeightsURL confirms the global -weights-url override
// still applies when fetchURL is nil and the requested model matches
// defaultModel — back-compat with pre-url.txt deployments.
func TestFetchHonoursLegacyWeightsURL(t *testing.T) {
	const wantURL = "https://example.com/legacy-default.onnx"
	s := New(
		nil,
		pb.VADModel_VAD_MODEL_PYANNOTE,
		func(m pb.VADModel) ([]byte, error) {
			return []byte("should-not-be-returned"), nil
		},
		nil, // no per-model URL lookup
		wantURL,
	)
	resp, err := s.Fetch(context.Background(), &pb.FetchRequest{
		Model: pb.VADModel_VAD_MODEL_PYANNOTE,
	})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if got := resp.GetUrl(); got != wantURL {
		t.Errorf("got url %q, want %q", got, wantURL)
	}
}

// TestFetchUnspecifiedModelUsesDefault confirms that an unset model field
// resolves to the server's defaultModel before URL/bytes lookup. (Same
// behaviour as before the url.txt change — checked here so a refactor
// doesn't accidentally bypass the resolution.)
func TestFetchUnspecifiedModelUsesDefault(t *testing.T) {
	const wantURL = "https://example.com/default-model.onnx"
	s := New(
		nil,
		pb.VADModel_VAD_MODEL_FIRERED,
		nil,
		func(m pb.VADModel) (string, bool) {
			if m == pb.VADModel_VAD_MODEL_FIRERED {
				return wantURL, true
			}
			return "", false
		},
		"",
	)
	// VAD_MODEL_UNSPECIFIED → server should treat as FIRERED.
	resp, err := s.Fetch(context.Background(), &pb.FetchRequest{})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if got := resp.GetUrl(); got != wantURL {
		t.Errorf("got url %q, want %q", got, wantURL)
	}
}
