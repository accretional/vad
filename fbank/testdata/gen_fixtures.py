#!/usr/bin/env python3
"""Regenerate kaldi-native-fbank ground-truth fixtures used by fbank_test.go.

Runs Kaldi's reference fbank impl (via the `kaldi_native_fbank` Python
binding) on a fixed audio sample with each of the configurations the Go
fbank package needs to match exactly. Output is raw float32 LE; the Go test
reads the same bytes and compares element-wise.

Run with any venv that has kaldi_native_fbank + numpy:
    /Volumes/wd_office_1/repos/firered-vad-bench/.venv/bin/python gen_fixtures.py

Each fixture file: `<window>-<frame_ms>-<shift_ms>-<mels>-<preemph>.f32` plus
a `.meta` JSON with the exact options used.
"""
import json
from pathlib import Path

import numpy as np
from kaldi_native_fbank import FbankOptions, OnlineFbank

HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent.parent / "data" / "sorry-dave-16k.f32"


def gen(label: str, window_type: str, preemph: float, scale: float = 1.0) -> None:
    """Run knf and save the [T, 80] float32 matrix.

    `scale` matches how upstream callers feed audio: FunASR/FireRed both
    multiply float32 [-1,1] PCM by 32768 before handing it to knf. The Go
    impl expects the same scaling at its input.
    """
    opts = FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.dither = 0
    opts.frame_opts.preemph_coeff = preemph
    opts.frame_opts.window_type = window_type
    opts.frame_opts.snip_edges = True
    opts.frame_opts.remove_dc_offset = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = 0  # sample_rate / 2

    fb = OnlineFbank(opts)
    samples_f32 = np.fromfile(AUDIO, dtype=np.float32)
    fb.accept_waveform(16000, (samples_f32 * scale).astype(np.float32))
    fb.input_finished()
    n = fb.num_frames_ready
    frames = np.stack([fb.get_frame(i) for i in range(n)]).astype(np.float32)
    out_path = HERE / f"{label}.f32"
    frames.tofile(out_path)
    meta = {
        "label": label,
        "audio": str(AUDIO.relative_to(HERE.parent.parent)),
        "input_scale": scale,
        "frame_opts": {
            "samp_freq": 16000,
            "frame_length_ms": 25.0,
            "frame_shift_ms": 10.0,
            "dither": 0,
            "preemph_coeff": preemph,
            "window_type": window_type,
            "snip_edges": True,
            "remove_dc_offset": True,
        },
        "mel_opts": {"num_bins": 80, "low_freq": 20, "high_freq": 0},
        "shape": list(frames.shape),
    }
    (HERE / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_path}  shape={frames.shape}")


if __name__ == "__main__":
    # Two configs cover both backends we'll wire up:
    #   povey + preemph 0.97 → FireRedVAD's expected features (raw int16-scaled PCM)
    #   hamming + preemph 0.97 → FSMN-VAD's expected features (also int16-scaled)
    gen("povey-25-10-80-pre097-scale32768", "povey", 0.97, scale=32768.0)
    gen("hamming-25-10-80-pre097-scale32768", "hamming", 0.97, scale=32768.0)
