#!/usr/bin/env python3
"""Export FunASR fsmn-vad (HF: funasr/fsmn-vad) to a standalone ONNX graph.

Run with the fsmn-vad-bench venv (which already has FunASR + torch):

    HF_HOME=/Volumes/wd_office_1/hf-cache \
    /Volumes/wd_office_1/repos/fsmn-vad-bench/.venv/bin/python \
        /Volumes/wd_office_1/repos/speax/benchmarks/vad/export_fsmn_vad_to_onnx.py

This downloads model.pt + am.mvn + config.yaml from HuggingFace, instantiates the
FunASR `FsmnVADStreaming` model, swaps its encoder for the export-friendly
`FSMNExport` wrapper (one ONNX-traceable forward, no python-side state
machine), traces it with `torch.onnx.export` at opset 14, and validates
PyTorch vs ONNX-Runtime parity on `data/ref-10s.wav`.

================================================================================
WHAT THE ONNX GRAPH ACTUALLY DOES (this is the *encoder only* — VAD state
machine stays in user code)
================================================================================

INPUTS
------
  speech       float32 [B, T_lfr, 400]     log-mel-fbank after LFR + CMVN
  in_cache0    float32 [B, 128, 19, 1]     per-layer FSMN ring buffer (zeros for first chunk)
  in_cache1    float32 [B, 128, 19, 1]
  in_cache2    float32 [B, 128, 19, 1]
  in_cache3    float32 [B, 128, 19, 1]

OUTPUTS
-------
  logits       float32 [B, T_lfr, 248]     softmax over 248 phone-like states
                                           sil_pdf_ids=[0]: `logits[..., 0]` is
                                           the *silence* probability; speech
                                           score = 1 - logits[..., 0].
  out_cache0   float32 [B, 128, 19, 1]     updated FSMN cache to feed next chunk
  out_cache1   float32 [B, 128, 19, 1]
  out_cache2   float32 [B, 128, 19, 1]
  out_cache3   float32 [B, 128, 19, 1]

(Cache shape: proj_dim=128, (lorder-1)*lstride = 19, 1)

DYNAMIC AXES: batch (always 1 in practice) and `feats_length` (T_lfr) on
`speech` and on every output. Cache axes are static.

================================================================================
PREPROCESSING the Go side must implement (raw PCM -> `speech`)
================================================================================

  1. Audio in: 16 kHz mono float32 in [-1, 1].  If int16, divide by 32768.
     Resample to 16 kHz first if needed.

  2. Kaldi fbank (torchaudio.compliance.kaldi.fbank defaults; FunASR overrides
     dither=0 and uses a Hamming window).  Exact params:
         sample_frequency = 16000
         num_mel_bins     = 80
         frame_length     = 25 ms   (400 samples)
         frame_shift      = 10 ms   (160 samples)
         window_type      = "hamming"
         dither           = 0.0
         snip_edges       = true    (kaldi default; drops trailing partial frame)
         energy_floor     = 0       (kaldi default)
         preemph_coeff    = 0.97    (kaldi default)
         remove_dc_offset = true    (kaldi default)
         htk_compat       = false   (kaldi default)
         use_energy       = false   (kaldi default)
         raw_energy       = true    (kaldi default — unused since use_energy=false)
         low_freq         = 20      (kaldi default)
         high_freq        = 0       (kaldi default -> Nyquist)
     Output shape: [T, 80].  IMPORTANT: kaldi fbank scales audio internally by
     32768; if you feed float in [-1,1] you may want to multiply by 32768 to
     match the training distribution.  (FunASR's WavFrontend does NOT
     pre-scale — it relies on torchaudio.compliance.kaldi to handle that.
     `torchaudio.compliance.kaldi.fbank` keeps float input in [-1,1] but
     internally scales as if int16.  See parity numbers below — Go-side must
     reproduce torchaudio/kaldi behaviour bit-exactly to within ~1e-3.)

  3. LFR (low-frame-rate) stacking with lfr_m=5, lfr_n=1:
         out[t] = concat(fbank[t-2 .. t+2])        (80*5 = 400 dims)
         Left edge: replicate fbank[0] twice as left padding.
         Right edge: replicate fbank[T-1] until last full window is reached.
     Output shape: [T_lfr, 400].  In offline / non-streaming mode T_lfr == T.

  4. CMVN from am.mvn:
         Parse the file once.  Two arrays of length 400:
             means  = <AddShift>/<LearnRateCoef>   (the row of 400 negative numbers)
             rescales = <Rescale>/<LearnRateCoef>  (the row of 400 small positives)
         Per frame:
             x = (x + means) * rescales              (note: + means, since
                                                     `means` is already negated
                                                     in the file)
         Output shape unchanged: [T_lfr, 400].

  5. Add batch dim -> [1, T_lfr, 400].  Feed as `speech`.  Feed zeros of
     shape [1, 128, 19, 1] as in_cache0..3 for one-shot (offline) inference;
     for streaming feed the previous chunk's out_cache0..3.

================================================================================
POSTPROCESSING (ONNX output -> [(start_s, end_s)] speech segments)
================================================================================

The graph emits per-frame state probabilities.  The VAD decision is a finite
state machine over those probabilities — see
`funasr.models.fsmn_vad_streaming.model.FsmnVADStreaming` for the canonical
implementation.  The minimum viable pipeline:

  a. silence_prob[t]   = logits[0, t, 0]
     speech_prob[t]    = 1 - silence_prob[t]
     If speech_prob[t] >= silence_prob[t] + speech_noise_thres  (default 0.6),
     mark frame t as SPEECH, else SIL.

  b. Smooth with a 200 ms sliding window (`window_size_ms=200`,
     `frame_in_ms=10` -> 20-frame window).  Transitions:
       SIL -> SPEECH if window-sum of SPEECH frames >= 15 (sil_to_speech_time_thres / 10)
       SPEECH -> SIL if window-sum of SPEECH frames <=  15
     Then apply `max_end_silence_time` (default 800 ms) hangover and the
     lookback / lookahead extends (`lookback_time_start_point=200`,
     `lookahead_time_end_point=100`) to convert state-transitions into
     [start_ms, end_ms] segments.

  c. Each frame is 10 ms; segment timestamps are emitted in ms.

For Go: port `WindowDetector` + `DetectOneFrame` from the python source —
they are deterministic and have no PyTorch dependencies.

================================================================================
FILES WRITTEN
================================================================================
  /Volumes/wd_office_1/repos/vad/weights/fsmn-vad/model.onnx
  /Volumes/wd_office_1/repos/vad/weights/fsmn-vad/am.mvn
  /Volumes/wd_office_1/repos/vad/weights/fsmn-vad/config.yaml
  /Volumes/wd_office_1/repos/speax/benchmarks/out/fsmn-vad/onnx_parity.txt
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# Keep weight caches off the home volume (per task spec).
os.environ.setdefault("HF_HOME", "/Volumes/wd_office_1/hf-cache")

import numpy as np
import soundfile as sf
import torch
import yaml
from huggingface_hub import hf_hub_download
from scipy import signal as sp_signal

# FunASR internals
from funasr.frontends.wav_frontend import WavFrontend, apply_cmvn, apply_lfr, load_cmvn
from funasr.models.fsmn_vad_streaming.model import FsmnVADStreaming
from funasr.models.fsmn_vad_streaming.export_meta import (
    export_dummy_inputs,
    export_dynamic_axes,
    export_forward,
    export_input_names,
    export_output_names,
)
from funasr.models.fsmn_vad_streaming.encoder import FSMNExport

import torchaudio.compliance.kaldi as kaldi  # to mirror WavFrontend exactly

REPO_ID = "funasr/fsmn-vad"
OUT_DIR = Path("/Volumes/wd_office_1/repos/vad/weights/fsmn-vad")
PARITY_DIR = Path("/Volumes/wd_office_1/repos/speax/benchmarks/out/fsmn-vad")
REF_WAV = Path("/Volumes/wd_office_1/repos/speax/benchmarks/data/ref-10s.wav")
OPSET = 14  # FunASR's reference export uses 14; opset 17 also works


def download_hf_files() -> dict[str, Path]:
    """Pull all four files from HF."""
    out = {}
    for fn in ("config.yaml", "configuration.json", "am.mvn", "model.pt"):
        out[fn] = Path(hf_hub_download(repo_id=REPO_ID, filename=fn))
    return out


def load_pt_model(config_yaml: Path, model_pt: Path) -> tuple[FsmnVADStreaming, dict]:
    """Build FsmnVADStreaming and load model.pt weights, mirroring AutoModel."""
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    # Build the model from config (mirror funasr.auto.auto_model._build_model)
    model = FsmnVADStreaming(
        encoder=cfg["encoder"],
        encoder_conf=cfg["encoder_conf"],
        **cfg["model_conf"],
    )

    state = torch.load(str(model_pt), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}", file=sys.stderr)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def install_export_methods(model: FsmnVADStreaming, cfg: dict):
    """Swap encoder for export-friendly version and bind helpers (mirrors
    FunASR's `export_rebuild_model`)."""
    import types

    model.encoder = FSMNExport(model.encoder, onnx=True)
    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)


def preprocess_audio(wav_path: Path, cmvn_path: Path, cfg: dict) -> torch.Tensor:
    """Replicate WavFrontendOnline(offline-mode) preprocessing end-to-end.

    Returns float32 tensor of shape [1, T_lfr, 400].
    """
    pcm, sr = sf.read(str(wav_path), dtype="float32")
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    if sr != cfg["frontend_conf"]["fs"]:
        target = cfg["frontend_conf"]["fs"]
        pcm = sp_signal.resample_poly(pcm, target, sr)
        sr = target

    # Match WavFrontend.forward exactly (it scales by 32768 before fbank)
    waveform = torch.from_numpy(pcm).float().unsqueeze(0) * 32768.0

    fc = cfg["frontend_conf"]
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=fc["n_mels"],
        frame_length=fc["frame_length"],
        frame_shift=fc["frame_shift"],
        dither=fc.get("dither", 0.0),
        energy_floor=0.0,
        window_type=fc.get("window", "hamming"),
        sample_frequency=fc["fs"],
    )  # [T, 80]

    # LFR
    if fc.get("lfr_m", 1) != 1 or fc.get("lfr_n", 1) != 1:
        mat = apply_lfr(mat, fc["lfr_m"], fc["lfr_n"])  # [T_lfr, 400]

    # CMVN
    cmvn = load_cmvn(str(cmvn_path))
    mat = apply_cmvn(mat, cmvn)  # [T_lfr, 400]

    return mat.unsqueeze(0)  # [1, T_lfr, 400]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "model.onnx"

    # 1. Download
    files = download_hf_files()
    shutil.copy2(files["am.mvn"], OUT_DIR / "am.mvn")
    shutil.copy2(files["config.yaml"], OUT_DIR / "config.yaml")

    # 2. Build PyTorch model
    pt_model, cfg = load_pt_model(files["config.yaml"], files["model.pt"])

    # 3. Get a real preprocessed sample for the trace (improves graph robustness)
    feats = preprocess_audio(REF_WAV, files["am.mvn"], cfg)
    print(f"[preprocess] feats: shape={tuple(feats.shape)} dtype={feats.dtype}")

    # 4. Run a *full* PyTorch reference forward through the unmodified encoder
    pt_encoder_ref = pt_model.encoder  # original FSMN (not export wrapper)
    with torch.no_grad():
        ref_logits = pt_encoder_ref(feats.clone(), cache=None)  # [1, T_lfr, 248]
    print(f"[pt-ref] logits: shape={tuple(ref_logits.shape)}")

    # 5. Install export hooks (swaps encoder for FSMNExport) and trace
    install_export_methods(pt_model, cfg)

    # Build matching dummy cache (zeros) for the trace.
    enc_conf = cfg["encoder_conf"]
    cache_frames = enc_conf["lorder"] + enc_conf["rorder"] - 1  # 19
    proj_dim = enc_conf["proj_dim"]  # 128
    zero_cache = tuple(
        torch.zeros(1, proj_dim, cache_frames, 1) for _ in range(enc_conf["fsmn_layers"])
    )

    # Run export-wrapper forward once to confirm parity with reference
    with torch.no_grad():
        exp_logits, exp_caches = pt_model(feats.clone(), *zero_cache)
    diff_pt = (exp_logits - ref_logits).abs().max().item()
    print(f"[pt-export vs pt-ref] max-abs diff: {diff_pt:.2e}")
    assert diff_pt < 1e-6, "Export-wrapper diverges from reference encoder"

    # 6. Export
    # Use the legacy TorchScript-based exporter (dynamo=False) — the new
    # dynamo path on torch 2.12 chokes on the (Tensor, [Tensor*4]) return
    # tree when passed `dynamic_axes` instead of `dynamic_shapes`, and our
    # graph is small enough that we don't need the dynamo features.
    dynamic_axes = {
        "speech": {1: "feats_length"},
        "logits": {1: "feats_length"},
    }
    tmp_path = Path("/tmp/fsmn-vad.onnx")
    torch.onnx.export(
        pt_model,
        (feats, *zero_cache),
        str(tmp_path),
        input_names=["speech", "in_cache0", "in_cache1", "in_cache2", "in_cache3"],
        output_names=[
            "logits",
            "out_cache0",
            "out_cache1",
            "out_cache2",
            "out_cache3",
        ],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    shutil.move(str(tmp_path), str(onnx_path))
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"[onnx] wrote {onnx_path} ({size_mb:.2f} MB)")

    # 7. Validate with onnxruntime
    import onnx
    import onnxruntime as ort

    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feats_np = feats.numpy()
    zero_cache_np = [c.numpy() for c in zero_cache]
    ort_inputs = {
        "speech": feats_np,
        "in_cache0": zero_cache_np[0],
        "in_cache1": zero_cache_np[1],
        "in_cache2": zero_cache_np[2],
        "in_cache3": zero_cache_np[3],
    }
    # Warm + timed
    _ = sess.run(None, ort_inputs)
    t0 = time.perf_counter()
    ort_outs = sess.run(None, ort_inputs)
    ort_ms = (time.perf_counter() - t0) * 1000

    ort_logits = ort_outs[0]
    diff_onnx = float(np.abs(ort_logits - ref_logits.numpy()).max())
    print(f"[onnx vs pt-ref] max-abs diff: {diff_onnx:.3e}")
    print(f"[onnx] sample latency (10s clip, CPU): {ort_ms:.1f} ms")

    # 8. Report
    in_meta = {i.name: (i.shape, i.type) for i in sess.get_inputs()}
    out_meta = {o.name: (o.shape, o.type) for o in sess.get_outputs()}

    parity_pass = diff_onnx < 1e-3
    lines = []
    lines.append(f"repo:           {REPO_ID}")
    lines.append(f"onnx path:      {onnx_path}")
    lines.append(f"onnx size:      {size_mb:.2f} MB")
    lines.append(f"opset:          {OPSET}")
    lines.append(f"parity passed:  {parity_pass}  (threshold 1e-3)")
    lines.append(f"max-abs diff:   {diff_onnx:.3e}  (pt export wrapper vs onnx)")
    lines.append(f"pt-ref vs pt-export diff: {diff_pt:.3e}")
    lines.append(f"sample input feats shape: {tuple(feats.shape)} dtype=float32")
    lines.append(f"onnxruntime latency (10s clip, CPU EP, warm): {ort_ms:.1f} ms")
    lines.append("")
    lines.append("INPUTS:")
    for name, (shape, dtype) in in_meta.items():
        lines.append(f"  {name:12s}  shape={shape}  dtype={dtype}")
    lines.append("")
    lines.append("OUTPUTS:")
    for name, (shape, dtype) in out_meta.items():
        lines.append(f"  {name:12s}  shape={shape}  dtype={dtype}")
    lines.append("")
    lines.append("Sample softmax row (frame 0):")
    lines.append(f"  silence_prob = logits[0,0,0] = {float(ort_logits[0,0,0]):.4f}")
    lines.append(f"  speech_prob  = 1 - logits[0,0,0] = {1.0 - float(ort_logits[0,0,0]):.4f}")
    report = "\n".join(lines)
    print()
    print(report)
    (PARITY_DIR / "onnx_parity.txt").write_text(report + "\n")
    print(f"\n[report] saved to {PARITY_DIR / 'onnx_parity.txt'}")

    if not parity_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
