# tests/stream

Validation client for the `DetectStream` bidi gRPC RPC. Streams a wav file to a running `vad` server in fixed-size chunks (default 100 ms) and prints each `SegmentationEvent` as it arrives.

## What it covers

- End-to-end streaming latency — how long after audio arrives does an activity transition or segment event come back? Look at `wall_s - audio_s` in the output.
- Activity-transition behaviour — are `speech_active` flips clean or jittery?
- Completed-segment quality — boundaries vs unary `Detect` output, dedupe, speaker assignment.

## Run

```bash
# Easiest: use a wav from the bundled data/ dir
bash tests/stream/run.sh

# Specify a wav + override defaults
bash tests/stream/run.sh /path/to/your.wav --chunk-ms 50 --max-seconds 10

# Against a remote server
bash tests/stream/run.sh /path/to/your.wav --addr 192.168.1.5:50051
```

`run.sh` regenerates Python stubs into `/tmp` on each invocation (no committed generated code), so the only Python prereqs are `grpcio`, `grpcio-tools`, `numpy`, `soundfile`, `scipy`.

Override the Python interpreter with `PYTHON=...` (e.g., `PYTHON=/path/to/.venv/bin/python bash tests/stream/run.sh ...`).

## Sample output

```
input: data/sorry-dave-16k.wav (10.00 s @ 16000 Hz)
connecting to localhost:50051, chunk=100 ms, realtime=True

 wall_s  audio_s  event
------------------------------------------------------------
   3.71     3.60  activity -> SPEECH
   4.53     4.40  activity -> silent
   4.96     4.80  activity -> SPEECH
   5.37     1.46  segment 1.17..1.46s speaker=0 conf=0.44
   ...
------------------------------------------------------------
events: 6 activity, 25 segment; total wall 15.56s
```

`wall_s` is wall-clock time since the client opened the stream; `audio_s` is the timestamp the server attached to the event (seconds into the input audio).

## Known scaffold quirks (tracked, not blocking)

The current server impl is a rolling-buffer sliding-window approach over the unary `Detect` backend. As a result you may see:

- Multiple `segment` events for the same speech region with slight boundary jitter — the rolling window re-runs inference and Pyannote sometimes shifts boundaries by a few frames. The server deduplicates by exact start time so near-duplicates leak through. Fix: quantize the dedupe key to ~100 ms.
- Short `silent -> SPEECH` flicker during the gap between adjacent Pyannote segments belonging to the same utterance. Fix: debounce activity transitions with a configurable hold time.

Per-backend native streaming (FSMN's `chunk_size`, FireRedVAD Stream-VAD variant) is a future optimization that would replace the sliding-window loop with an incremental inference call.
