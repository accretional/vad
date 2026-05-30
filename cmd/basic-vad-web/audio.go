// audio.go: HTTP handlers that proxy the speax/audio MediaConverter gRPC
// service. Two responsibilities:
//
//   1. POST /upload    — receive any container ffmpeg understands, decode to
//                        16 kHz mono float32 WAV via ConversionStream, return
//                        the WAV bytes to the browser so the existing decode
//                        path (Web Audio decodeAudioData) can pick it up
//                        unchanged. Removes the browser's hard dependency on
//                        Web Audio supporting the container (Safari + .mkv,
//                        Firefox + .m4a, etc.).
//
//   2. GET  /svg       — generate a waveform SVG for a sample / uploaded clip
//                        by calling AudioToVectors + VectorsToSvg back-to-back.
//                        The browser drops the SVG straight into the DOM
//                        instead of computing the waveform itself.
//
// We talk to a separate audio gRPC backend (default localhost:50052). The vad
// backend on :50051 is unchanged.

package main

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	audioclient "ffmpegapi/pkg/client"
)

// audioApp holds the connection + cache state for the audio-server proxy.
// Embedded into app via main.go so handlers can reach it as a.audio.*.
type audioApp struct {
	addr   string
	client *audioclient.Client

	svgMu    sync.Mutex
	svgCache map[string]string // key = file-id|width|height → svg xml
}

func newAudioApp(addr string) (*audioApp, error) {
	c, err := audioclient.New(addr)
	if err != nil {
		return nil, fmt.Errorf("dial audio %s: %w", addr, err)
	}
	return &audioApp{
		addr:     addr,
		client:   c,
		svgCache: make(map[string]string),
	}, nil
}

// ---------------------------------------------------------------------------
// /upload  — multipart media file → 16 kHz mono float32 WAV bytes
// ---------------------------------------------------------------------------

// handleUpload accepts a multipart form with a single "file" field of any
// container ffmpeg can decode. It writes the bytes to a temp file, asks the
// audio server to transcode to a WAV with the standard VAD-friendly layout
// (16 kHz mono, f32le samples), and streams the resulting WAV back as
// application/octet-stream. The browser side feeds those bytes into the same
// decodeAudioData path it uses for bundled samples.
//
// Why WAV and not raw PCM? The frontend already wraps Float32Arrays in a WAV
// header for preview playback; returning a self-describing WAV here lets us
// reuse the existing decode pipeline verbatim. We pick f32le inside the WAV
// so no quantisation happens — the bytes flow through to VAD exactly as
// ffmpeg produced them.
// handleUploadOr503 / handleSvgOr503 gate the audio-backed endpoints on the
// audio client being configured. We want the demo to keep serving the rest of
// the UI when -audio-addr is empty (or the audio server is down at startup).
func (a *app) handleUploadOr503(w http.ResponseWriter, r *http.Request) {
	if a.audio == nil {
		http.Error(w, "audio backend not configured (set -audio-addr)", http.StatusServiceUnavailable)
		return
	}
	a.handleUpload(w, r)
}

func (a *app) handleSvgOr503(w http.ResponseWriter, r *http.Request) {
	if a.audio == nil {
		http.Error(w, "audio backend not configured (set -audio-addr)", http.StatusServiceUnavailable)
		return
	}
	a.handleSvg(w, r)
}

func (a *app) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	// 200 MB cap — well above any reasonable VAD demo clip.
	if err := r.ParseMultipartForm(200 << 20); err != nil {
		http.Error(w, "parse multipart: "+err.Error(), http.StatusBadRequest)
		return
	}
	f, hdr, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing file field: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer f.Close()

	tmpDir, err := os.MkdirTemp("", "vad-upload-*")
	if err != nil {
		http.Error(w, "mktmp: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.RemoveAll(tmpDir)

	// Preserve the extension so ffmpeg picks the right demuxer.
	ext := filepath.Ext(hdr.Filename)
	if ext == "" {
		ext = ".bin"
	}
	inPath := filepath.Join(tmpDir, "input"+ext)
	pcmPath := filepath.Join(tmpDir, "output.pcm")
	dst, err := os.Create(inPath)
	if err != nil {
		http.Error(w, "create temp: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := io.Copy(dst, f); err != nil {
		dst.Close()
		http.Error(w, "save upload: "+err.Error(), http.StatusInternalServerError)
		return
	}
	dst.Close()

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()
	if err := a.audio.client.ConvertToRawPCM(ctx, inPath, pcmPath, audioclient.AudioRawOptions{
		SampleRate:   16000,
		Channels:     1,
		SampleFormat: "f32le",
	}); err != nil {
		http.Error(w, "audio convert: "+err.Error(), http.StatusBadGateway)
		return
	}

	pcm, err := os.ReadFile(pcmPath)
	if err != nil {
		http.Error(w, "read converted: "+err.Error(), http.StatusInternalServerError)
		return
	}
	// Wrap the raw float32 PCM in a WAV header so the browser's existing
	// decodeAudioData path can pick it up unchanged. f32le inside WAV uses
	// format tag 3 (IEEE float).
	data := wrapWavFloat32(pcm, 16000, 1)
	// Persist the converted WAV so the audio server can re-read it for /svg
	// later. We stash it under a stable path keyed by the response hash.
	outWavPath := filepath.Join(tmpDir, "output.wav")
	_ = os.WriteFile(outWavPath, data, 0644)

	// Tag the response with a stable file id so the browser can ask /svg for
	// the matching waveform without re-uploading. The id is the sha1 of the
	// converted bytes — deterministic, lets us memoise per-clip SVGs across
	// retries of the same file.
	sum := sha1.Sum(data)
	id := "u-" + hex.EncodeToString(sum[:8])
	a.audio.cacheUpload(id, outWavPath, data)

	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("X-Audio-Id", id)
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	_, _ = w.Write(data)
}

// ---------------------------------------------------------------------------
// /svg  — waveform SVG via the audio gRPC service
// ---------------------------------------------------------------------------

// handleSvg returns a waveform SVG for a known clip. The clip is identified by
// `?id=` (returned by /upload, or one of the bundled-sample ids "s-<name>").
// `?w=` / `?h=` are optional dimensions; default 900x80. Result is cached
// per (id, w, h) tuple in-process — the bundled samples are warmed at startup.
func (a *app) handleSvg(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "missing id", http.StatusBadRequest)
		return
	}
	width := parseDim(r.URL.Query().Get("w"), 900)
	height := parseDim(r.URL.Query().Get("h"), 80)

	cacheKey := fmt.Sprintf("%s|%d|%d", id, width, height)
	a.audio.svgMu.Lock()
	if svg, ok := a.audio.svgCache[cacheKey]; ok {
		a.audio.svgMu.Unlock()
		w.Header().Set("Content-Type", "image/svg+xml")
		w.Header().Set("Cache-Control", "public, max-age=300")
		_, _ = io.WriteString(w, svg)
		return
	}
	a.audio.svgMu.Unlock()

	path, ok := a.audio.pathFor(id, a.weightsRoot)
	if !ok {
		http.Error(w, "unknown id "+id, http.StatusNotFound)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	vectors, err := a.audio.client.AudioToVectors(ctx, path, 0)
	if err != nil {
		http.Error(w, "audio_to_vectors: "+err.Error(), http.StatusBadGateway)
		return
	}
	resp, err := a.audio.client.VectorsToSvg(ctx, vectors, int32(width), int32(height))
	if err != nil {
		http.Error(w, "svg: "+err.Error(), http.StatusBadGateway)
		return
	}
	svg := resp.GetSvg().GetContent()

	a.audio.svgMu.Lock()
	a.audio.svgCache[cacheKey] = svg
	a.audio.svgMu.Unlock()

	w.Header().Set("Content-Type", "image/svg+xml")
	w.Header().Set("Cache-Control", "public, max-age=300")
	_, _ = io.WriteString(w, svg)
}

// wrapWavFloat32 returns a WAVE/RIFF container around a raw little-endian
// float32 PCM stream. Format tag = 3 (IEEE float), 32 bits per sample.
// We hand-roll the header because (a) it's 44 bytes, (b) we'd otherwise pull
// in an entire audio library for one struct write.
func wrapWavFloat32(pcm []byte, sampleRate, channels int) []byte {
	dataSize := uint32(len(pcm))
	byteRate := uint32(sampleRate) * uint32(channels) * 4
	blockAlign := uint16(channels) * 4
	bitsPerSample := uint16(32)

	var buf bytes.Buffer
	buf.Grow(44 + len(pcm))
	buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(36+len(pcm)))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))   // PCM subchunk size
	_ = binary.Write(&buf, binary.LittleEndian, uint16(3))    // IEEE float
	_ = binary.Write(&buf, binary.LittleEndian, uint16(channels))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(&buf, binary.LittleEndian, byteRate)
	_ = binary.Write(&buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(&buf, binary.LittleEndian, bitsPerSample)
	buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, dataSize)
	buf.Write(pcm)
	return buf.Bytes()
}

func parseDim(raw string, def int) int {
	if raw == "" {
		return def
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 || v > 4096 {
		return def
	}
	return v
}

// ---------------------------------------------------------------------------
// Upload <-> path mapping + bundled-sample resolution
// ---------------------------------------------------------------------------

// uploadEntry tracks a converted upload on disk so subsequent /svg calls can
// re-read the bytes without involving the browser. We keep the path (instead
// of the bytes) so memory doesn't grow unbounded across many uploads; an LRU
// would be overkill for a single-user demo, so we just retain the last N.
type uploadEntry struct {
	path    string
	addedAt time.Time
}

var (
	uploadMu      sync.Mutex
	uploadByID    = make(map[string]*uploadEntry)
	uploadHistory []string
	uploadKeep    = 8 // bounded — drop oldest when we exceed.
)

func (au *audioApp) cacheUpload(id, path string, _ []byte) {
	// Copy the converted file into a stable temp path so the per-request temp
	// dir from handleUpload can be cleaned up without losing the data we need
	// for /svg. We don't keep the bytes in memory.
	stableDir := filepath.Join(os.TempDir(), "vad-uploads")
	_ = os.MkdirAll(stableDir, 0755)
	stable := filepath.Join(stableDir, id+".wav")
	if src, err := os.Open(path); err == nil {
		defer src.Close()
		if dst, err := os.Create(stable); err == nil {
			_, _ = io.Copy(dst, src)
			dst.Close()
		}
	}

	uploadMu.Lock()
	defer uploadMu.Unlock()
	if _, exists := uploadByID[id]; !exists {
		uploadHistory = append(uploadHistory, id)
		for len(uploadHistory) > uploadKeep {
			old := uploadHistory[0]
			uploadHistory = uploadHistory[1:]
			if oe, ok := uploadByID[old]; ok {
				_ = os.Remove(oe.path)
				delete(uploadByID, old)
			}
		}
	}
	uploadByID[id] = &uploadEntry{path: stable, addedAt: time.Now()}
}

// pathFor resolves a clip id ("u-..." upload, or "s-<basename>" sample) to a
// filesystem path the audio server can read. Bundled samples live under
// static/samples/ in the same repo and are written out to a temp directory
// once on startup (samplesDir).
func (au *audioApp) pathFor(id, weightsRoot string) (string, bool) {
	switch {
	case len(id) > 2 && id[:2] == "u-":
		uploadMu.Lock()
		defer uploadMu.Unlock()
		if oe, ok := uploadByID[id]; ok {
			return oe.path, true
		}
	case len(id) > 2 && id[:2] == "s-":
		name := id[2:]
		if p := samplePath(name); p != "" {
			return p, true
		}
	}
	return "", false
}

// ---------------------------------------------------------------------------
// Bundled sample disk extraction (so the audio server can read them)
// ---------------------------------------------------------------------------

var samplesOnDiskMu sync.Mutex
var samplesOnDisk = make(map[string]string) // basename → path

// extractBundledSamples writes every static/samples/* file to a temp dir so
// the audio server (a separate process, no access to our go:embed FS) can
// read them by path. Called once at startup; idempotent.
func extractBundledSamples() error {
	samplesOnDiskMu.Lock()
	defer samplesOnDiskMu.Unlock()
	if len(samplesOnDisk) > 0 {
		return nil
	}
	tmp, err := os.MkdirTemp("", "vad-samples-*")
	if err != nil {
		return err
	}
	entries, err := staticFS.ReadDir("static/samples")
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		data, err := staticFS.ReadFile("static/samples/" + e.Name())
		if err != nil {
			return err
		}
		dest := filepath.Join(tmp, e.Name())
		if err := os.WriteFile(dest, data, 0644); err != nil {
			return err
		}
		samplesOnDisk[e.Name()] = dest
	}
	return nil
}

func samplePath(name string) string {
	samplesOnDiskMu.Lock()
	defer samplesOnDiskMu.Unlock()
	return samplesOnDisk[name]
}

// warmSampleSvgs precomputes the default-size SVG for each bundled sample so
// the very first page load doesn't show empty waveforms while AudioToVectors
// is grinding. Failures are logged but non-fatal (the audio server may not be
// up yet during tests; /svg will lazily retry on demand).
func (a *app) warmSampleSvgs() {
	samplesOnDiskMu.Lock()
	names := make([]string, 0, len(samplesOnDisk))
	for n := range samplesOnDisk {
		names = append(names, n)
	}
	samplesOnDiskMu.Unlock()

	for _, n := range names {
		id := "s-" + n
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		vecs, err := a.audio.client.AudioToVectors(ctx, samplePath(n), 0)
		if err != nil {
			log.Printf("warm svg %s: AudioToVectors: %v", n, err)
			cancel()
			continue
		}
		resp, err := a.audio.client.VectorsToSvg(ctx, vecs, 900, 80)
		if err != nil {
			log.Printf("warm svg %s: VectorsToSvg: %v", n, err)
			cancel()
			continue
		}
		key := fmt.Sprintf("%s|%d|%d", id, 900, 80)
		a.audio.svgMu.Lock()
		a.audio.svgCache[key] = resp.GetSvg().GetContent()
		a.audio.svgMu.Unlock()
		cancel()
	}
}
