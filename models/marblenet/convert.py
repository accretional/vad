#!/usr/bin/env python3
"""Export NVIDIA Frame_VAD_Multilingual_MarbleNet_v2.0 to a standalone ONNX graph.

Run with the marblenet-bench venv (which has nemo_toolkit[asr] + onnxruntime):

    HF_HOME=/Volumes/wd_office_1/hf-cache \
    /Volumes/wd_office_1/repos/marblenet-bench/.venv/bin/python \
        /Volumes/wd_office_1/repos/speax/benchmarks/vad/export_marblenet_to_onnx.py

================================================================================
WHAT THE ONNX GRAPH ACTUALLY DOES (encoder + decoder; preprocessor stays in
user code)
================================================================================

INPUTS
------
  audio_signal   float32 [B, 80, T_mel]    log-Mel spectrogram (channels-first!)
                                           Both B and T_mel are dynamic axes
                                           (named "audio_signal_dynamic_axes_1"
                                           and "audio_signal_dynamic_axes_2"
                                           by the NeMo exporter).

OUTPUTS
-------
  outputs        float32 [B, T_out, 2]     RAW LOGITS (NOT log-probs!) over
                                           {non-speech, speech}.
                                           Apply softmax to get probabilities:
                                              p_speech = softmax(out)[..., 1]
                                           T_out = T_mel / 2 (the encoder
                                           subsamples by 2; for 10 ms mel
                                           frames this means one logit per
                                           20 ms — matches the model card).

DYNAMIC AXES
------------
  Batch (always 1 in practice) and T_mel are dynamic. The model is a CNN
  (MarbleNet); padding is constant, so different T_mel just give different
  T_out frames with no recurrent state to carry between calls.

================================================================================
PREPROCESSING the Go side must implement (raw PCM -> `audio_signal`)
================================================================================

  1. Audio in: 16 kHz mono float32 in [-1, 1]. Resample from source SR first
     if needed.

  2. Log-mel spectrogram with the following Kaldi-ish params (these come
     from the .nemo's preprocessor config:
        _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
        normalize: None
        window_size: 0.025         (25 ms)
        sample_rate: 16000
        window_stride: 0.01        (10 ms)
        window: hann
        features: 80
        n_fft: 512
        frame_splicing: 1
        dither: 1.0e-05            (SET TO 0 FOR DETERMINISM)
        pad_to: 2

     Implementation notes (matching NeMo's FilterbankFeatures):
        - HOP_LEN = window_stride * sample_rate = 160 samples
        - WIN_LEN = window_size  * sample_rate = 400 samples
        - n_fft   = 512   (zero-padded to next power of 2 above WIN_LEN)
        - window  = hann(400, periodic=True)
        - PRE-EMPHASIS: x[t] -= 0.97 * x[t-1]  (default preemph=0.97)
        - PAD: pad_to=2 right-pads time so T_mel is a multiple of 2 (zero
          padding on the time axis of the spectrogram, AFTER STFT).
        - MEL FILTERBANK: 80 banks, slaney scale, fmin=0, fmax=8000
          (sample_rate/2). Matches `librosa.filters.mel(sr=16000, n_fft=512,
          n_mels=80, fmin=0, fmax=8000, norm='slaney', htk=False)`.
        - LOG: out = log(mel_spec + 2**-24)  (NeMo's default `log_zero_guard
          _value` is `2**-24`; `log_zero_guard_type='add'`).
        - NORMALIZE: None (do NOT apply per-feature or per-utterance CMVN —
          the model was trained without it).

     Final shape: [1, 80, T_mel] where
        T_mel ≈ ceil((num_samples + 1) / 160)   (NeMo uses center-padded
        STFT with reflection padding; for 16 000 samples this gives 1002,
        rounded UP to a multiple of 2 by pad_to=2.)

  3. Feed as audio_signal to ONNX. No "length" input is exported — the model
     was built with the (no-cache) export_for_inference path, which folds the
     full T into the encoder's "max_audio_length" (masking is turned off at
     export time).

================================================================================
POSTPROCESSING (ONNX output -> [(start_s, end_s)] speech segments)
================================================================================

The graph emits per-frame logits at 50 Hz (one per 20 ms). The recommended
recipe (mirrors NVIDIA's `vad_utils.generate_vad_segment_table`):

  a. probs[t] = softmax(logits[t])[1]
  b. Optional smoothing: median filter (e.g. width 5 frames = 100 ms) or
     boxcar mean.
  c. Threshold: speech if probs[t] >= onset (default 0.5); leave when
     probs[t] < offset (e.g. 0.3) — hysteresis.
  d. State machine with minimum durations:
        min_duration_on  = 200 ms (10 frames) — drop short speech runs
        min_duration_off = 100 ms ( 5 frames) — merge short silences
  e. (Optional) pad start/end by ~50–100 ms each side.
  f. Emit (start_s, end_s) with frame_idx * 0.02 timestamps.

================================================================================
FILES WRITTEN
================================================================================
  /Volumes/wd_office_1/repos/vad/weights/marblenet/model.onnx
  /Volumes/wd_office_1/repos/vad/weights/marblenet/preprocessor.yaml
  /Volumes/wd_office_1/repos/speax/benchmarks/out/marblenet-onnx/onnx_parity.txt
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import warnings
from pathlib import Path

# Keep weight caches off the home volume.
os.environ.setdefault("HF_HOME", "/Volumes/wd_office_1/hf-cache")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import omegaconf  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from scipy import signal as sp_signal  # noqa: E402

import nemo.collections.asr as nemo_asr  # noqa: E402
from nemo.collections.asr.parts.submodules.jasper import MaskedConv1d  # noqa: E402

HF_REPO = "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0"
HF_FILE = "frame_vad_multilingual_marblenet_v2.0.nemo"
OUT_DIR = Path("/Volumes/wd_office_1/repos/vad/weights/marblenet")
PARITY_DIR = Path("/Volumes/wd_office_1/repos/speax/benchmarks/out/marblenet-onnx")
REF_WAV = Path("/Volumes/wd_office_1/repos/speax/benchmarks/data/ref-10s.wav")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "model.onnx"

    # 1. Pull the .nemo checkpoint
    print(f"[hf] downloading {HF_REPO}/{HF_FILE}")
    nemo_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    print(f"[hf] -> {nemo_path}")

    # 2. Load the model
    print("[nemo] restoring EncDecFrameClassificationModel ...")
    # strict=False: the .nemo state_dict is missing `loss.weight` (a buffer)
    model = nemo_asr.models.EncDecFrameClassificationModel.restore_from(
        nemo_path, map_location="cpu", strict=False
    )
    model.eval()
    # Determinism: kill the dither noise the preprocessor adds by default.
    model.preprocessor.featurizer.dither = 0.0
    for p in model.parameters():
        p.requires_grad_(False)

    # 3. Save the preprocessor config (so the Go side has the source of truth).
    pre_cfg = omegaconf.OmegaConf.to_yaml(model.cfg.preprocessor)
    (OUT_DIR / "preprocessor.yaml").write_text(pre_cfg)
    print(f"[save] preprocessor.yaml -> {OUT_DIR / 'preprocessor.yaml'}")

    # 4. Compute a reference feature set on the 10 s clip (parity input).
    pcm, sr = sf.read(str(REF_WAV), dtype="float32")
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    if sr != 16000:
        pcm = sp_signal.resample_poly(pcm, 16000, sr)
    audio = torch.from_numpy(pcm).unsqueeze(0)
    audio_len = torch.tensor([pcm.shape[0]], dtype=torch.long)
    with torch.no_grad():
        proc, proc_len = model.preprocessor(input_signal=audio, length=audio_len)
    print(f"[preprocess] mel: shape={tuple(proc.shape)} dtype={proc.dtype}")

    # 5. PyTorch reference forward — match the export-time graph state:
    #    masked convolutions OFF, eval mode. NeMo's exporter applies the same
    #    transforms via `_prepare_for_export`.
    for m in model.modules():
        if isinstance(m, MaskedConv1d):
            m.use_mask = False
    with torch.no_grad():
        ref_logits = model.forward_for_export(input=proc, length=proc_len)
    print(f"[pt-ref] logits: shape={tuple(ref_logits.shape)}")

    # 6. Export via NeMo's built-in helper (this internally calls
    #    torch.onnx.export with sensible dynamic axes + input_example).
    print(f"[onnx] exporting -> {onnx_path}")
    model.export(str(onnx_path), check_trace=False)

    # 7. Parity check vs onnxruntime
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    ort_logits = sess.run([out_name], {in_name: proc.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(ref_logits.numpy() - ort_logits)))
    mean_abs_diff = float(np.mean(np.abs(ref_logits.numpy() - ort_logits)))
    print(f"[parity] max-abs diff = {max_abs_diff:.3e}, mean = {mean_abs_diff:.3e}")

    # 8. Quick warm + timed run
    for _ in range(2):
        sess.run([out_name], {in_name: proc.numpy()})
    N = 5
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run([out_name], {in_name: proc.numpy()})
    onnx_ms = (time.perf_counter() - t0) * 1000.0 / N

    in_shape = sess.get_inputs()[0].shape
    in_dtype = sess.get_inputs()[0].type
    out_shape = sess.get_outputs()[0].shape
    out_dtype = sess.get_outputs()[0].type
    onnx_size = onnx_path.stat().st_size

    # Sample softmax for sanity
    logits_np = ort_logits[0]  # [T_out, 2]
    probs = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    speech_prob = probs[:, 1]

    lines = []
    lines.append(f"repo:           {HF_REPO}")
    lines.append(f"checkpoint:     {HF_FILE}")
    lines.append(f"onnx path:      {onnx_path}")
    lines.append(f"onnx size:      {onnx_size/1024:.1f} KiB")
    lines.append(f"params:         {sum(p.numel() for p in model.parameters())}")
    lines.append(f"sample audio:   {REF_WAV} ({pcm.shape[0]/16000:.3f} s)")
    lines.append(f"mel input shape: {tuple(proc.shape)} dtype=float32 (B, 80, T_mel)")
    lines.append(f"logits shape:    {tuple(ref_logits.shape)} dtype=float32 (B, T_out, 2)")
    lines.append(f"frame rate (model output): ~{ref_logits.shape[1] / (pcm.shape[0]/16000):.2f} Hz")
    lines.append(f"parity max-abs diff: {max_abs_diff:.3e}  (threshold 1e-3)")
    lines.append(f"parity mean-abs diff: {mean_abs_diff:.3e}")
    lines.append(f"parity (<1e-3): {'PASS' if max_abs_diff < 1e-3 else 'FAIL'}")
    lines.append(f"onnxruntime: {ort.__version__} (CPUExecutionProvider)")
    lines.append(f"latency / call (10 s clip, warm, N={N}): {onnx_ms:.2f} ms")
    lines.append("")
    lines.append("ONNX INPUTS:")
    lines.append(f"  {in_name}  shape={in_shape}  dtype={in_dtype}")
    lines.append("ONNX OUTPUTS:")
    lines.append(f"  {out_name}  shape={out_shape}  dtype={out_dtype}")
    lines.append("")
    lines.append("Sample frame logits:")
    lines.append(f"  frame[0]   = {logits_np[0].tolist()}")
    lines.append(f"  frame[T/2] = {logits_np[len(logits_np)//2].tolist()}")
    lines.append(f"speech prob mean (softmax): {float(speech_prob.mean()):.4f}")
    text = "\n".join(lines) + "\n"

    (PARITY_DIR / "onnx_parity.txt").write_text(text)
    print()
    print(text)
    print(f"[report] -> {PARITY_DIR / 'onnx_parity.txt'}")

    return 0 if max_abs_diff < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
