#!/usr/bin/env python3
"""Generate NeMo-style log-mel-spectrogram ground-truth fixtures for melspec_test.go.

Uses the existing marblenet-bench venv (which has nemo_toolkit[asr] +
torchaudio installed) to compute the reference output that the Go melspec
package must match. We instantiate AudioToMelSpectrogramPreprocessor with the
exact config saved alongside the MarbleNet ONNX export so the fixture is
identical to what the MarbleNet model sees in production.

Run:
    HF_HOME=/Volumes/wd_office_1/hf-cache \\
    /Volumes/wd_office_1/repos/marblenet-bench/.venv/bin/python gen_fixtures.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/Volumes/wd_office_1/hf-cache")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent.parent / "data" / "sorry-dave-16k.f32"


def make_preprocessor():
    # Avoid pulling all of NeMo just to instantiate one module — use the
    # underlying FilterbankFeatures directly. Mirrors NeMo's
    # AudioToMelSpectrogramPreprocessor with the MarbleNet config exactly.
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
    return FilterbankFeatures(
        sample_rate=16000,
        n_window_size=400,    # 25 ms
        n_window_stride=160,  # 10 ms
        window="hann",
        normalize=None,
        n_fft=512,
        preemph=0.97,
        nfilt=80,
        lowfreq=0,
        highfreq=8000,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=0,              # deterministic
        pad_to=2,
        frame_splicing=1,
        stft_exact_pad=False,
        stft_conv=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
    )


def main():
    fb = make_preprocessor()
    fb.eval()

    pcm = np.fromfile(AUDIO, dtype=np.float32)
    audio = torch.from_numpy(pcm).unsqueeze(0)  # [B=1, T]
    seq_len = torch.tensor([pcm.shape[0]])

    with torch.no_grad():
        feats, out_len = fb(audio, seq_len)
    # feats: [B=1, n_mels=80, T_mel]
    feats = feats.squeeze(0).numpy().astype(np.float32)  # [80, T_mel]
    # We save as [T_mel, 80] (row-major time-first) to match the Go
    # melspec.Compute output convention.
    feats_tf = feats.T.copy()  # [T_mel, 80]

    out_path = HERE / "nemo-sorry-dave.f32"
    feats_tf.tofile(out_path)
    meta = {
        "shape_time_first": list(feats_tf.shape),
        "shape_channels_first": list(feats.shape),
        "source_audio": str(AUDIO.relative_to(HERE.parent.parent)),
        "out_len_frames": int(out_len.item()),
        "preprocessor": {
            "sample_rate": 16000,
            "n_window_size": 400,
            "n_window_stride": 160,
            "window": "hann",
            "normalize": None,
            "n_fft": 512,
            "preemph": 0.97,
            "nfilt": 80,
            "lowfreq": 0,
            "highfreq": 8000,
            "log_zero_guard_value": 2 ** -24,
            "log_zero_guard_type": "add",
            "dither": 0,
            "pad_to": 2,
            "mel_norm": "slaney",
        },
    }
    (HERE / "nemo-sorry-dave.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_path}  shape={feats_tf.shape}  bytes={feats_tf.nbytes}")


if __name__ == "__main__":
    main()
