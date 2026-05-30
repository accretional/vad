# basic-vad-web

Browser demo for the multi-backend vad gRPC service. Lets you upload / record
/ pick a sample clip and compare segmentation output across five VAD
backends side-by-side **running entirely in the browser**, plus a live-streaming
panel that bridges your mic to `DetectStream` over a WebSocket against one
backing gRPC server.

## Architecture

```
                           +--------------------------------+
                           |  basic-vad-web (Go)            |
                           |  HTTP :8080                    |
                           |                                |
   Browser  <--/describe-- |  /describe  /fetch  /aux/...   |
                           |  /socket    /upload  /svg      |
            <--/static/--- |  /static/...                   |
                           +-+----------------+-------------+
                             |                |
                  gRPC :50051|                |gRPC :50052
                             v                v
                  +------------------+   +----------------------+
                  |  ./bin/vad       |   |  speax/audio server  |
                  |  (one VAD        |   |  (MediaConverter:    |
                  |  backend loaded; |   |   ConversionStream,  |
                  |  weights +       |   |   AudioToVectors,    |
                  |  url.txt         |   |   svg, ...)          |
                  |  embedded)       |   +----------------------+
                  +------------------+

                              In the browser (per backend, per Web Worker):

  static/app.js  →  static/js/engine.js
                       │
                       ├── new Worker('/static/js/worker.js')
                       │       loads onnxruntime-web from CDN
                       │       imports backends/<NAME>.js (preprocess/postprocess)
                       │       imports dsp/{fbank,melspec,fft}.js (port of Go code)
                       │
                       └── IndexedDB cache for model.onnx + aux sidecars
```

**THREE Go processes**:

1. `./bin/vad` (gRPC :50051) — one VAD backend loaded. Serves model weights
   (`Fetch`) and bridges the live mic (`DetectStream`).
2. `audio-server` (gRPC :50052) — the speax/audio `MediaConverter` service
   built from a sibling `../speax/audio` checkout. Used for decoding
   arbitrary container formats (mp4, mov, webm, m4a, flac, ...) to the
   16 kHz mono float32 WAV that the VAD pipeline expects, and for rendering
   waveform SVGs (`AudioToVectors` + `Svg`).
3. The demo HTTP server in front of both. The browser does all VAD inference
   for the side-by-side comparison; the gRPC servers are only consulted for
   weights, live mic, media decode, and waveform rendering.

## Running it

`run.sh` spawns three processes: vad (:50051), audio server (:50052), and the
demo HTTP server (:8080). It expects a sibling `../speax/audio` checkout (the
audio server is `go build`'d from there); override with `AUDIO_REPO=`. All
three processes are torn down on Ctrl-C.

```bash
# Build the vad gRPC server (one-time; ships every backend embedded).
./build-native.sh                # → ./bin/vad

# From the repo root, spawn vad + audio + demo HTTP server + open browser.
./cmd/basic-vad-web/run.sh                       # pyannote on :50051 (default)
./cmd/basic-vad-web/run.sh --backend silero      # any single vad backend
PORT=9090 ./cmd/basic-vad-web/run.sh --backend marblenet
AUDIO_PORT=50066 ./cmd/basic-vad-web/run.sh      # change audio gRPC port
AUDIO_REPO=/path/to/speax/audio ./cmd/basic-vad-web/run.sh
```

If the audio server isn't reachable the demo still boots — only `/upload` and
`/svg` return 503; bundled samples + in-browser inference + live mic keep
working.

The choice of vad backend only affects the live-streaming panel
(`/socket → DetectStream` runs against whatever backend the server loaded).
The batch comparison panel runs all five backends in the browser regardless
— their weights are pulled via `Fetch` on first use (URL redirect when the
backend has a `url.txt` sidecar; raw bytes streamed otherwise), then cached
in IndexedDB so subsequent reloads are instant.

## Browser inference pipeline

| Backend | Pre | Model | Post | JS file |
|---|---|---|---|---|
| pyannote | raw float32, 10 s windows | ONNX | argmax over 7 powerset classes → per-speaker segments | `static/js/backends/pyannote.js` |
| fsmn | Kaldi fbank (Hamming, 80 mel) → LFR m=5 → CMVN | ONNX | silence-prob threshold + sliding-window smoothing + hangover | `static/js/backends/fsmn.js` |
| firered | Kaldi fbank (Povey, 80 mel) → CMVN | ONNX | boxcar smooth → 4-state hysteresis | `static/js/backends/firered.js` |
| silero | raw float32, 512-sample chunks | ONNX (chunked state) | two-state hysteresis | `static/js/backends/silero.js` |
| marblenet | NeMo log-mel (Hann, Slaney) | ONNX | softmax → onset/offset hysteresis | `static/js/backends/marblenet.js` |

Each backend's JS pipeline ports the same logic from the matching
`pkg/vad/<name>.go`. Shared DSP lives under `static/js/dsp/` (Kaldi fbank,
NeMo melspec, FFT — direct ports of `fbank/` and `melspec/` in Go).

ONNX Runtime Web is loaded into each worker from jsDelivr
(`https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.bundle.min.mjs`).
The wasm binary itself is fetched once per browser session from the same
CDN and cached by the HTTP cache. If you want fully-offline operation,
swap the `ORT_URL` constant in `static/js/worker.js` for a path under
`static/vendor/`.

## What the page does

1. On load, `GET /describe` returns the list of `VADModel` enum values, the
   gRPC server's address, and (via reflection) the RPC method names.
2. Pick an audio source — bundled sample, file upload, or mic recording.
   Browser decodes to 16 kHz mono Float32Array via the Web Audio API.
3. Tick one or more backends, hit "Run detection" → for each selected
   backend, the engine lazily spawns a Worker, downloads the model on
   first use (via `/fetch` → URL redirect when available, else gRPC byte
   stream), persists it to IndexedDB, and runs inference. Subsequent runs
   reuse the warm worker.
4. Results render on a canvas with a shared time axis: waveform on top,
   one row per backend below with color-coded segment rectangles.
   Click a segment to play just that range.
5. Live panel: pick a backend with `--backend` when starting the demo,
   then click "Start mic" → opens a WebSocket to `/socket`, which bridges
   to the loaded server-side backend's `DetectStream`. Mic is captured in
   ~100 ms float32 chunks; activity transitions render as a speech
   indicator, finalized segments append to a log.

## HTTP / WS endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves `static/index.html` |
| `GET /static/...` | Embedded CSS/JS modules + bundled `samples/*.mp3` |
| `GET /describe` | JSON: service name, RPC methods (from reflection if available), known backends, gRPC server address |
| `GET /fetch?model=NAME` | Proxies the gRPC `Fetch` RPC. Returns `{"url": "..."}` (application/json) when the server has a CDN URL for the backend (via `url.txt` sidecar), otherwise the raw .onnx bytes (application/octet-stream) |
| `GET /aux/<dir>/<file>` | Serves auxiliary files (`am.mvn`, `cmvn_*.f32`, ...) from the same embedded weights tree the gRPC server uses. Allowlisted; no path traversal. |
| `WS /socket` | Bridges browser WebSocket ↔ `DetectStream` against the single backing gRPC backend. Binary frames = raw float32-LE PCM @ 16 kHz mono. Server sends JSON text events: `{type: "activity", speech_active, timestamp}` or `{type: "segment", start, end, speaker_id, confidence, timestamp}` |
| `POST /upload` | Multipart media file (`file` field) → 16 kHz mono float32 WAV. Proxies to the speax/audio server's `ConversionStream` (decode via `Transform.audio_raw` → ffmpeg `-ar 16000 -ac 1 -f f32le`) and re-wraps the raw PCM in a WAV header so the browser's existing `decodeAudioData` path can handle it unchanged. Response header `X-Audio-Id` returns a stable id usable with `/svg`. Accepts mp4, mov, m4a, webm, flac, mkv, ogg, opus, aac, plus the wav/mp3 already handled in-browser. Returns 503 when the audio backend is disabled. |
| `GET /svg?id=<id>&w=<W>&h=<H>` | Waveform SVG for a known clip (`u-…` upload returned from `/upload`, or `s-<basename>` for a bundled sample). Proxies to the audio server's `AudioToVectors` + `Svg` RPCs. Cached per `(id, w, h)` in-process; bundled samples are warmed at startup. Defaults: 900×80. Returns 503 when the audio backend is disabled. |

There's no `/detect` HTTP endpoint anymore — the browser does that work
locally via the Worker engine. If you want server-side inference for a
debugging cross-check, call the gRPC `Detect` RPC directly (e.g. via the
`cmd/pkg-example` snippet or grpcurl).

## url.txt convention

Each `weights/<backend>/` directory may contain a `url.txt` file with
exactly one HTTPS URL pointing at where `model.onnx` lives (typically
the same path on GitHub raw). The sidecar is embedded via `go:embed`
alongside the weights themselves (see `prep-embed.sh`).

When present, the `Fetch` RPC returns `FetchResponse.url = <that URL>`
instead of streaming the bytes; this is much friendlier for browser
clients (saves ~1-20 MB of gRPC payload per backend per first-time
visitor). Embed-first ordering matches the rest of the weight resolution:
embedded `url.txt` wins, then on-disk, then fall back to raw bytes.

## Files

```
cmd/basic-vad-web/
  main.go              HTTP server: /describe, /fetch, /aux, /socket
  audio.go             /upload, /svg + audio-gRPC client wiring + sample warmup
  main_test.go         /describe shape, /aux allowlist, static FS contents
  run.sh               Spawns vad + audio-server + demo HTTP, opens browser
  static/
    index.html         Two-section page: batch compare + live mic
    app.js             ES module: source picker, fanout to engine, canvas render
    style.css
    samples/
      bestfriends.mp3
      sorry-dave.mp3
      wake-me-up.mp3
    js/
      engine.js        Manages one Worker per backend, IDB cache, /fetch
      worker.js        Per-backend Worker: loads ORT, runs inference
      cache.js         IndexedDB wrapper
      dsp/
        fft.js         Radix-2 Cooley-Tukey (port of fbank/fft.go)
        fbank.js       Kaldi-style log-mel fbank (port of fbank/)
        melspec.js     NeMo log-mel (port of melspec/)
      backends/
        pyannote.js    powerset decoding per 10-s window
        fsmn.js        LFR-stacked fbank + CMVN + smoothing
        firered.js     fbank + CMVN + 4-state hysteresis
        silero.js      chunked-state inference + two-state hysteresis
        marblenet.js   NeMo melspec + 2-class softmax + onset/offset
```

## Known gaps

- **ORT loaded from CDN**, not vendored. The WASM blob is ~11 MB, which
  felt heavy to bundle into a small demo. The CDN ships wide CORS so it
  works from any origin; first-load latency is one CDN round-trip per
  unique browser. For a fully offline build, replace the `ORT_URL`
  constant in `static/js/worker.js` with a path under `static/vendor/`.
- **AudioWorklet** would be the modern replacement for `ScriptProcessorNode`
  in the live panel's mic capture. Works fine on every current browser,
  just deprecated.
- **Pyannote powerset decoding** is fully implemented in JS (not a stub).
  Cross-window merging is identical to the Go path.
