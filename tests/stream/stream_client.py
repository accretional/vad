#!/usr/bin/env python3
"""Validation client for the DetectStream bidi RPC.

Streams an audio file to the running vad server in fixed-size chunks (defaults
to 100 ms — close to what a live mic callback would produce) and prints each
SegmentationEvent it receives, including wall-clock and audio-time deltas.
Lets you spot:
  - end-to-end streaming latency (wall-vs-audio gap on each event),
  - duplicate / jittery activity transitions,
  - completed-segment quality vs the unary Detect output.

Pre-reqs:
  - vad server listening on --addr (default localhost:50051)
  - Python: grpcio, numpy, soundfile, scipy
  - Python stubs vad_pb2.py / vad_pb2_grpc.py in cwd or PYTHONPATH

The simplest way to run is via `tests/stream/run.sh`, which generates stubs to
/tmp and pipes them in. To run by hand:

  python -m grpc_tools.protoc --proto_path=proto --python_out=/tmp --grpc_python_out=/tmp proto/vad.proto
  PYTHONPATH=/tmp python tests/stream/stream_client.py path/to/audio.wav
"""
import argparse
import sys
import time
from pathlib import Path

import grpc
import numpy as np
import soundfile as sf
from scipy import signal

import vad_pb2
import vad_pb2_grpc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="Audio file to stream (any sample rate; resampled to 16 kHz mono).")
    p.add_argument("--addr", default="localhost:50051", help="vad gRPC address")
    p.add_argument("--chunk-ms", type=int, default=100, help="Streaming chunk size in ms")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Truncate input to this many seconds (useful for long files)")
    p.add_argument("--realtime", action="store_true", default=True,
                   help="Sleep between chunks to simulate live mic capture (default on)")
    p.add_argument("--no-realtime", dest="realtime", action="store_false",
                   help="Send all chunks as fast as possible (stress test)")
    args = p.parse_args()

    data, sr = sf.read(args.audio, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = signal.resample_poly(data, 16000, sr)
        sr = 16000
    if args.max_seconds is not None:
        data = data[: int(sr * args.max_seconds)]
    data = data.astype(np.float32)
    print(f"input: {args.audio} ({len(data) / sr:.2f} s @ {sr} Hz)")

    chunk_samples = sr * args.chunk_ms // 1000

    def chunk_iter():
        for i in range(0, len(data), chunk_samples):
            chunk = data[i : i + chunk_samples]
            eos = (i + chunk_samples) >= len(data)
            yield vad_pb2.AudioChunk(
                samples=chunk.tobytes(),
                sample_rate=16000 if i == 0 else 0,
                end_of_stream=eos,
            )
            if args.realtime:
                time.sleep(args.chunk_ms / 1000.0)

    with grpc.insecure_channel(args.addr) as ch:
        stub = vad_pb2_grpc.VoiceSegmentationStub(ch)
        print(f"connecting to {args.addr}, chunk={args.chunk_ms} ms, realtime={args.realtime}")
        print()
        print(f"{'wall_s':>7}  {'audio_s':>7}  event")
        print("-" * 60)
        t0 = time.time()
        n_activity = n_segment = 0
        for event in stub.DetectStream(chunk_iter()):
            wall = time.time() - t0
            kind = event.WhichOneof("event")
            if kind == "activity":
                n_activity += 1
                state = "SPEECH" if event.activity.speech_active else "silent"
                print(f"{wall:>7.2f}  {event.timestamp:>7.2f}  activity -> {state}")
            elif kind == "segment":
                n_segment += 1
                s = event.segment
                print(f"{wall:>7.2f}  {event.timestamp:>7.2f}  segment {s.start:.2f}..{s.end:.2f}s "
                      f"speaker={s.speaker_id} conf={s.confidence:.2f}")
        print("-" * 60)
        print(f"events: {n_activity} activity, {n_segment} segment; total wall {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
