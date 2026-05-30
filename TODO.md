# TODO

Outstanding work in `accretional/vad`. Organized roughly by importance.

## Streaming RPC scaffolding

- [ ] **Dedupe completed-segment emissions** in `DetectStream`. Current `emittedStarts` key is exact start time (float64); Pyannote re-computes boundaries with ~ms-level jitter across windows, so the same segment is emitted multiple times. Quantize the key to ~100 ms buckets.
- [ ] **Debounce activity transitions**. Short gaps between adjacent Pyannote segments in the same utterance register as `silent → speech` flicker. Add a configurable hold time (e.g. 200 ms) — frames must agree for that long before flipping the activity flag.
- [ ] **Per-backend native streaming**. The scaffold re-runs unary `ProcessAudio` on a rolling buffer; native streaming would avoid wasted compute:
  - FSMN exports an `in_cache0..3` / `out_cache0..3` interface; we currently feed zero caches every call. Wire the cache loop through `DetectStream` so each chunk carries the previous chunk's caches.
  - FireRedTeam ships a separate Stream-VAD variant (`Stream-VAD/model.pth.tar` in the HF repo) explicitly designed for incremental inference — would need its own ONNX export.
  - Pyannote is window-based; no easy streaming path.

## Backend abstraction

- [ ] Add an optional `StreamBackend` interface (extends `Backend`) that backends implement when they have native streaming. `internal/server/server.go`'s `DetectStream` would use it if available; fall back to the sliding-window scaffold otherwise.
- [ ] Per-backend smoke tests in `tests/<backend>/` using the existing `tests/` convention (mirror `tests/stream/`).

## Build & deploy

- [ ] `cmd/vad/run.sh` for the native-build flow. Today the only "run" path is via Docker (per `build.sh`), and we currently use ad-hoc commands for native runs. A run script would keep the repo's "scripts only" convention.
- [ ] `setup.sh` could optionally re-export the FSMN / FireRed ONNX from source via the speax export scripts (avoiding the need to commit weights). Today they're bundled in `weights/{fsmn,firered}-vad/` (~4 MB total).
- [ ] CI: `go test ./...` against the bundled weights + fbank fixtures. The repo passes locally; no automated runner yet.

## Documentation

- [ ] Spec the `DetectStream` event semantics more precisely in the proto comments (current text is hand-wavy on whether `timestamp` is the start, end, or transition point of each event).
- [ ] `pkg/vad/README.md` (if the package grows) covering the `Backend` interface contract: what `Segment.SpeakerID` means for VAD-only backends, error semantics, lifecycle.
