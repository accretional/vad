# basic-vad-web

Browser demo for the multi-backend vad gRPC service. Lets you upload / record
/ pick a sample clip and compare segmentation output across multiple VAD
backends side-by-side, plus a live-streaming panel that bridges your mic to
`DetectStream` over a WebSocket.

## Why this demo runs *N* `./bin/vad` processes instead of one

This is the unintuitive bit, so read this first.

The vad server (`./bin/vad`) now **embeds all backends' weights and the ONNX
Runtime dylib via `go:embed`**, so a single binary ships everything needed
for pyannote / FSMN / FireRed / Silero / (future) MarbleNet. Good for
distribution — one binary, no setup script.

**However**, at *runtime* a single `./bin/vad` process still loads exactly
**one** backend (the one named by `-backend` or `VADConfig.model`). The
Detect / DetectStream RPCs unconditionally dispatch to that one backend;
there is no per-request `model` field on those RPCs today (only `Fetch`
takes a per-request model param, for pulling weights of arbitrary backends).

So to compare backends side-by-side, this demo connects to **multiple
`./bin/vad` instances** — one per backend, each on its own port. `run.sh`
spawns them for you (`--all` brings up all four working backends on
:50051-:50054); the demo HTTP server fans `/detect` requests out to whichever
addresses you passed via `-vad-addrs`. The result is functionally what you'd
want, but the topology is weird — one Go process per backend.

If/when the server grows per-request backend selection (a `model` field on
`Audio` / `AudioChunk` and a backend pool inside `internal/server/server.go`),
this demo can collapse to a single backing process. Until then, multi-process
is the cleanest workaround. The alternative — having the demo restart `bin/vad`
with a different `-backend` between requests — is fragile and slow (weights
reload on every switch).

## Why this isn't a transformers.js / onnxruntime-web demo

You might reasonably expect: "the server `Fetch`-es weights, the browser runs
ONNX in a Web Worker via transformers.js / onnxruntime-web, no gRPC fanout
needed." That'd be a nicer architecture for a public demo, but it isn't what
this demo does today.

The only browser-inference hook that exists right now is the `weights_url`
field on `VADConfig` + the `Fetch` RPC's URL-redirect response. That's a hint
for future browser clients — it doesn't actually wire up any JS-side
inference. To run inference in the browser you'd need:

- A JS preprocessor matching each Go backend's `ProcessAudio` byte-for-byte
  (log-Mel features for FSMN/FireRed/MarbleNet, raw float32 windows for
  pyannote/Silero), and
- A JS postprocessor for each (powerset decoding for pyannote, frame
  thresholding + smoothing for everyone else).

That's a significant rewrite of `pkg/vad/{pyannote,fsmn,firered,silero}.go`
into JS. For a phased migration, Silero is the easy starter — it's small,
the public Silero JS implementations exist to copy from, and the segmentation
post-processing is trivial.

For now, the demo runs inference server-side via gRPC. Browser-side inference
is on the list but not built.

## Running it

```bash
# Build the vad gRPC server (one-time; ships every backend embedded).
./build-native.sh                # → ./bin/vad

# From the repo root, spawn 4 vad backends + the demo HTTP server + open browser.
./cmd/basic-vad-web/run.sh --all

# Just pyannote (default):
./cmd/basic-vad-web/run.sh

# Custom subset:
./cmd/basic-vad-web/run.sh --backends pyannote,silero

# Non-default HTTP port:
PORT=9090 ./cmd/basic-vad-web/run.sh --all
```

Backends that aren't wired up show as greyed-out checkboxes in the UI; the
multi-select still works, you just can't tick the disabled ones.

## What the page does

1. On load, `GET /describe` returns the list of `VADModel` enum values and
   which backends are wired up (via `-vad-addrs`). Reflection is attempted
   against the default backend; success / failure is shown in the header.
2. Pick an audio source — bundled sample, file upload, or mic recording.
   Browser decodes to 16 kHz mono Float32Array via the Web Audio API.
3. Pick one or more backends, hit "Run detection" → `POST /detect`
   multipart. Server fans out one concurrent `Detect` RPC per backend and
   returns per-model segments + timing.
4. Results render on a canvas with a shared time axis: waveform on top,
   one row per backend below with color-coded segment rectangles.
   Click a segment to play just that range.
5. Live panel: pick a backend, click "Start mic" → `WS /socket?model=…`
   opens a WebSocket bridged to `DetectStream`. Mic is captured in ~100 ms
   float32 chunks, sent as binary frames; activity transitions render as a
   speech indicator, finalized segments append to a log.

## HTTP / WS endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves `static/index.html` |
| `GET /static/...` | Embedded CSS/JS + bundled `samples/*.mp3` |
| `GET /describe` | JSON: service name, RPC methods (from reflection if available), per-model availability + addresses, default model |
| `POST /detect` | Multipart: `audio=<file>`, optional `encoding=f32le`, repeated `model=<NAME>`. Fans out one `Detect` per requested model. Returns `{audio_duration_seconds, results: [{model, segments, elapsed_ms, error?}]}` |
| `WS /socket?model=<NAME>` | Bridges browser WebSocket ↔ `DetectStream` for one backend. Binary frames = raw float32-LE PCM @ 16 kHz mono. Server sends JSON text events: `{type: "activity", speech_active, timestamp}` or `{type: "segment", start, end, speaker_id, confidence, timestamp}` |

`/detect` audio decoding: if the client posts `encoding=f32le` (what the
browser does, since it's already decoded the audio), the server skips
ffmpeg. Otherwise the server shells out to `ffmpeg` to decode arbitrary
container formats to raw 16 kHz mono float32.

## Files

```
cmd/basic-vad-web/
  main.go            HTTP server: /describe, /detect, /socket
  main_test.go       parseVADAddrs, /describe shape, mocked /detect fanout
  run.sh             Spawns N vad backends + demo + opens browser
  static/
    index.html       Three-section page: service info, batch compare, live
    app.js           Vanilla JS; Web Audio decode + canvas render + WS client
    style.css
    samples/
      bestfriends.mp3
      sorry-dave.mp3
      wake-me-up.mp3
```

## Known gaps

- **MarbleNet backend** isn't wired in the upstream gRPC server yet (it
  fatal-errors on startup), so the UI greys it out even if you try to add
  it to `-vad-addrs`.
- **No browser-side inference** (see section above).
- **AudioWorklet** would be the modern replacement for `ScriptProcessorNode`
  in the live panel's mic capture. Works fine on every current browser,
  just deprecated.
- **Single backend per `bin/vad` process** (see section above).
