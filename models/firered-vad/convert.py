#!/usr/bin/env python3
"""Export FireRedTeam/FireRedVAD (DFSMN VAD variant) to ONNX and verify parity.

USAGE
-----
    HF_HOME=/Volumes/wd_office_1/hf-cache \
        /Volumes/wd_office_1/repos/firered-vad-bench/.venv/bin/python \
        /Volumes/wd_office_1/repos/speax/benchmarks/vad/export_firered_vad_to_onnx.py

OUTPUTS
-------
    /Volumes/wd_office_1/repos/vad/weights/firered-vad/model.onnx
    /Volumes/wd_office_1/repos/vad/weights/firered-vad/cmvn.ark            (raw kaldi cmvn for reference)
    /Volumes/wd_office_1/repos/vad/weights/firered-vad/cmvn_means.f32      (float32 80-d, little-endian)
    /Volumes/wd_office_1/repos/vad/weights/firered-vad/cmvn_istd.f32       (float32 80-d, little-endian)
    /Volumes/wd_office_1/repos/speax/benchmarks/out/firered-vad/onnx_parity.txt

GO RE-IMPLEMENTATION RECIPE (preprocessing + ONNX I/O + postprocessing)
-----------------------------------------------------------------------
Audio pipeline (must run before invoking the ONNX graph):

  1. Decode the source to 16 kHz mono PCM.
  2. Convert samples to **int16** in [-32768, 32767]. The kaldi-native-fbank
     extractor used upstream expects int16 PCM. (Floats in [-1,1] are
     silently treated as ~0 and produce useless features — confirmed in
     the FireRed bench.)
  3. Compute 80-channel log-Mel filterbank features with Kaldi conventions:
        sample_rate     = 16000 Hz
        frame_length    = 25 ms  (400 samples)
        frame_shift     = 10 ms  (160 samples)
        num_mel_bins    = 80
        dither          = 0.0
        snip_edges      = True   (frames covering only valid samples)
        window          = povey  (kaldi default — kaldi-native-fbank
                                  `FbankOptions` defaults match what
                                  fireredvad uses)
        energy_floor    = 0.0    (knf default; output is log-mel only)
     The resulting feature tensor has shape (T, 80) where
        T = floor((num_samples - 400) / 160) + 1   (snip_edges=True).
  4. Apply CMVN (cepstral mean / variance normalisation) per feature dim:
        feat[t, d] = (feat[t, d] - mean[d]) * inverse_std[d]
     mean[] and inverse_std[] are baked from cmvn.ark and stored alongside
     the ONNX as cmvn_means.f32 / cmvn_istd.f32 (raw little-endian
     float32, 80 values each). They were derived as:
        mean      = stats[0, d] / count
        variance  = stats[1, d] / count - mean*mean      (floor 1e-20)
        inv_std   = 1 / sqrt(variance)
  5. Reshape to (1, T, 80) float32 (batch axis = 1) and feed to ONNX.

ONNX graph
----------
    input  name="feat"   shape=[N, T, 80]   dtype=float32   (dynamic N, T)
    output name="probs"  shape=[N, T,  1]   dtype=float32   (sigmoid prob
                                                              of speech
                                                              per 10 ms frame)

Postprocessing (frame probabilities -> speech segments). This mirrors
FireRedVAD's `VadPostprocessor` with the **default** config from
`fireredvad/vad.py` (FireRedVadConfig dataclass):

    smooth_window_size   = 5      (boxcar smoothing of probabilities)
    speech_threshold     = 0.4    (post-smoothing threshold)
    min_speech_frame     = 20     (200 ms — state-machine hysteresis)
    max_speech_frame     = 2000   (20 s — long-segment splitter)
    min_silence_frame    = 20     (200 ms — state-machine hysteresis)
    merge_silence_frame  = 0      (disabled by default)
    extend_speech_frame  = 0      (disabled by default)

Algorithm:
    a) Smooth: boxcar mean over a length-5 window, with cumulative
       running mean for the first 4 samples (so the output length
       equals the input length).
    b) Threshold: smoothed_prob >= 0.4.
    c) State machine over {SILENCE, POSSIBLE_SPEECH, SPEECH,
       POSSIBLE_SILENCE} with min_speech_frame / min_silence_frame
       transitions (see `_smooth_preds_with_state_machine` in
       `fireredvad/core/vad_postprocessor.py` for exact transitions).
    d) Whenever a silence->speech transition happens, back-fill the
       previous `smooth_window_size` decisions to 1 (compensates the
       boxcar lag).
    e) (Optional) Merge short silences and extend speech segments.
    f) Split runs of speech longer than max_speech_frame at the
       local-min-probability point in the second half of each window.
    g) Emit (start_s, end_s) pairs with start_s = first_speech_frame *
       0.01 and end_s = first_silence_frame * 0.01 (last segment uses
       len(decisions)*0.01 + 0.025, capped to clip duration).

ARCHITECTURE CONSTRAINTS
------------------------
The model is a DFSMN (Deep Feed-forward Sequential Memory Network):
    - Pure feed-forward over time; processes a whole (B, T, 80) tensor.
    - No recurrent state in the non-stream graph (cache inputs/outputs
      are omitted; this is the "non-stream" variant identical to the
      upstream `fireredvad_vad.onnx`).
    - Lookahead is built into the FSMN layers (N2*S2 frames), so for a
      sequence of length T the output corresponds to the same T frames
      (causal-style padding internally).
    - Latency is roughly linear in T. Memory is modest (~24M params
      worth of activations are not produced; the model is ~600k params).
    - For streaming, use the upstream `fireredvad_stream_vad_with_cache`
      ONNX instead — that graph carries a list of cache tensors. This
      script intentionally exports the non-streaming variant.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Paths & environment
# ----------------------------------------------------------------------
os.environ.setdefault("HF_HOME", "/Volumes/wd_office_1/hf-cache")

FRV_REPO = Path("/Volumes/wd_office_1/repos/firered-vad-bench/FireRedVAD")
WAV = Path("/Volumes/wd_office_1/repos/speax/benchmarks/data/ref-10s.wav")
OUT_DIR = Path("/Volumes/wd_office_1/repos/vad/weights/firered-vad")
REPORT_PATH = Path(
    "/Volumes/wd_office_1/repos/speax/benchmarks/out/firered-vad/onnx_parity.txt"
)
ONNX_PATH = OUT_DIR / "model.onnx"

sys.path.insert(0, str(FRV_REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from fireredvad.core.detect_model import DetectModel  # noqa: E402
from fireredvad.core.audio_feat import AudioFeat, CMVN  # noqa: E402


# ----------------------------------------------------------------------
# Wrapper module — strips the cache outputs so the graph has a single
# clean input ("feat") and a single output ("probs").
# ----------------------------------------------------------------------
class FireRedVADNonStream(nn.Module):
    """Wrap DetectModel so torch.onnx.export sees a fixed (Tensor)->Tensor."""

    def __init__(self, detect_model: DetectModel):
        super().__init__()
        self.model = detect_model
        self.model.eval()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        probs, _ = self.model.forward(feat, caches=None)
        return probs


def fetch_model_dir() -> Path:
    snap = snapshot_download(
        repo_id="FireRedTeam/FireRedVAD",
        allow_patterns=["VAD/*", "config.yaml"],
    )
    return Path(snap)


def load_audio_int16(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = signal.resample_poly(data, 16000, sr)
    return np.clip(data * 32768.0, -32768, 32767).astype(np.int16)


def write_cmvn_binaries(cmvn: CMVN, out_dir: Path) -> None:
    cmvn.means.astype("float32").tofile(out_dir / "cmvn_means.f32")
    cmvn.inverse_std_variances.astype("float32").tofile(out_dir / "cmvn_istd.f32")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    snap = fetch_model_dir()
    vad_dir = snap / "VAD"
    print(f"snapshot: {snap}")
    print(f"vad_dir : {vad_dir}")

    # ------------------------------------------------------------------
    # 1. Copy cmvn + config to output dir, write float32 cmvn binaries
    # ------------------------------------------------------------------
    import shutil

    shutil.copyfile(vad_dir / "cmvn.ark", OUT_DIR / "cmvn.ark")
    src_cfg = snap / "config.yaml"
    if src_cfg.exists():
        shutil.copyfile(src_cfg, OUT_DIR / "config.yaml")
        print(f"copied config.yaml -> {OUT_DIR / 'config.yaml'}")
    else:
        print("WARN: config.yaml not present in snapshot")

    cmvn = CMVN(str(vad_dir / "cmvn.ark"))
    write_cmvn_binaries(cmvn, OUT_DIR)
    print(f"copied cmvn.ark + cmvn_*.f32 -> {OUT_DIR}")

    # ------------------------------------------------------------------
    # 2. Load PyTorch model
    # ------------------------------------------------------------------
    print("loading DetectModel...")
    detect = DetectModel.from_pretrained(str(vad_dir))
    detect.eval()
    wrapper = FireRedVADNonStream(detect)

    # ------------------------------------------------------------------
    # 3. Compute reference features on the 10 s clip
    # ------------------------------------------------------------------
    pcm_i16 = load_audio_int16(WAV)
    feat_extractor = AudioFeat(str(vad_dir / "cmvn.ark"))
    feat, dur = feat_extractor.extract(pcm_i16)  # (T, 80) float32 torch
    feat_t = feat.unsqueeze(0).contiguous()  # (1, T, 80)
    print(f"audio: {WAV.name} dur={dur:.3f}s feat={tuple(feat_t.shape)}")

    # ------------------------------------------------------------------
    # 4. Export ONNX (opset 17, dynamic batch + time)
    # ------------------------------------------------------------------
    print(f"exporting ONNX -> {ONNX_PATH}")
    # Use a non-trivial dummy length so the trace exercises FSMN lookahead.
    dummy = torch.randn(1, 100, 80, dtype=torch.float32)
    # Notes on exporter choice:
    #   * The legacy TorchScript exporter (dynamo=False) honours
    #     opset_version=17 exactly, but bakes the dummy time dimension
    #     (T=100) into FSMN's lookahead arithmetic — parity collapses
    #     on real inputs (max-abs diff ~0.2).
    #   * The dynamo exporter (default) traces with torch.export and
    #     preserves dynamic shapes correctly (parity 3e-7), but:
    #         - silently rejects opset 17 because the Pad op only has a
    #           v18 down-converter; output graph ends up at opset 18.
    #         - defaults to external_data=True (writes model.onnx and a
    #           sibling model.onnx.data). Set external_data=False so the
    #           Go service (yalue/onnxruntime_go) sees one self-contained
    #           file.
    # We use the dynamo path with external_data=False. The opset 18
    # graph is fully compatible with the onnxruntime versions current Go
    # bindings ship (>= 1.15 -> IR v9 / opset 18).
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(ONNX_PATH),
        input_names=["feat"],
        output_names=["probs"],
        opset_version=17,  # actual graph may be opset 18 (see note above)
        do_constant_folding=True,
        dynamic_axes={
            "feat": {0: "batch", 1: "time"},
            "probs": {0: "batch", 1: "time"},
        },
        dynamo=True,
        external_data=False,
    )

    # ------------------------------------------------------------------
    # 5. Parity check: PyTorch vs ONNX on the same feat
    # ------------------------------------------------------------------
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_probs = wrapper(feat_t).cpu().numpy()  # (1, T, 1)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    onnx_probs = sess.run([out_name], {in_name: feat_t.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_probs - onnx_probs)))
    mean_abs_diff = float(np.mean(np.abs(torch_probs - onnx_probs)))

    # Timing: average over a few runs after warm-up.
    for _ in range(2):
        sess.run([out_name], {in_name: feat_t.numpy()})
    N = 5
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run([out_name], {in_name: feat_t.numpy()})
    onnx_ms = (time.perf_counter() - t0) * 1000.0 / N

    onnx_size = ONNX_PATH.stat().st_size

    in_shape = sess.get_inputs()[0].shape
    in_dtype = sess.get_inputs()[0].type
    out_shape = sess.get_outputs()[0].shape
    out_dtype = sess.get_outputs()[0].type

    report = []
    report.append(f"FireRedVAD (DFSMN, non-stream) -> ONNX parity report")
    report.append(f"audio          : {WAV} ({dur:.3f} s)")
    report.append(f"feat shape     : {tuple(feat_t.shape)} (B, T, mel)")
    report.append(f"onnx file      : {ONNX_PATH} ({onnx_size/1024:.1f} KiB)")
    report.append(f"input          : name={in_name} shape={in_shape} dtype={in_dtype}")
    report.append(f"output         : name={out_name} shape={out_shape} dtype={out_dtype}")
    report.append(f"torch probs    : shape={torch_probs.shape} dtype={torch_probs.dtype}")
    report.append(f"onnx  probs    : shape={onnx_probs.shape} dtype={onnx_probs.dtype}")
    report.append(f"max abs diff   : {max_abs_diff:.3e}")
    report.append(f"mean abs diff  : {mean_abs_diff:.3e}")
    report.append(f"parity (<1e-3) : {'PASS' if max_abs_diff < 1e-3 else 'FAIL'}")
    report.append(f"onnxruntime    : {ort.__version__} (CPUExecutionProvider)")
    report.append(f"latency / call : {onnx_ms:.2f} ms over {N} runs on 10 s clip")
    text = "\n".join(report) + "\n"

    REPORT_PATH.write_text(text)
    print()
    print(text)
    print(f"report written -> {REPORT_PATH}")

    return 0 if max_abs_diff < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
