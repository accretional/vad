# TODO

Outstanding work in `accretional/vad`. Organized roughly by importance.

## New backends

- [x] ~~Silero VAD backend~~ — done. `pkg/vad/silero.go`. 14 segs / 27.8s on ref clip; per-chunk state threading; bundled ONNX at `weights/silero/model.onnx` (1.26 MiB).
- [ ] **MarbleNet backend** — `nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`. Benched + ONNX exported (361 KiB, **3.8 ms/10 s** — the fastest of all backends, ~4× faster than FireRed). Multilingual (zh/en/fr/es/de/ru). Go backend not yet written because it requires a separate feature pipeline distinct from `fbank/` (Kaldi-style):
  - **Window**: Hann (not Povey/Hamming).
  - **STFT**: `n_fft=512`, `hop=160`, `win_length=400`. center=True with reflection padding on the waveform (different from our snip_edges=true convention).
  - **Pre-emphasis**: 0.97 on the whole waveform before framing (not per-frame).
  - **Mel filterbank**: Slaney scale, 80 bins, 0-8000 Hz (Kaldi uses HTK scale + low-cutoff at 20 Hz).
  - **Log**: `log(mel + 2^-24)` (not `log(max(mel, eps))`).
  - **Pad to even**: trailing time-axis padding so `T_mel` is divisible by 2.
  - **Postprocess**: `softmax(logits)[..., 1]` → per-20ms speech prob → onset/offset hysteresis (0.5/0.3 + min_duration_on=200ms / min_duration_off=100ms).
  Cleanest path is probably extending `fbank/Options` with `Window=Hann`, `MelType=Slaney`, `PadMode=Center`, `LogOffset` fields. Parity-test against a fixture generated via the existing `marblenet-bench` venv (`/Volumes/wd_office_1/repos/marblenet-bench/.venv` has `nemo_toolkit[asr]` already installed; preprocessor config saved at `weights/marblenet/preprocessor.yaml`). Bench script: `speax/benchmarks/vad/bench_marblenet_onnx.py`. Estimated 2-3 hours focused work.

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
