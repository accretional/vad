#!/usr/bin/env python3
"""Export Silero VAD (16 kHz) to ONNX from the upstream TorchScript model.

UPSTREAM SOURCE
---------------
The PyTorch JIT file ships inside the `silero-vad` PyPI package
(https://pypi.org/project/silero-vad/), which is the same artifact
github.com/snakers4/silero-vad serves via `torch.hub.load('snakers4/silero-vad',
'silero_vad', onnx=False)`. We use the pip package because the torch.hub path
fails on hosts without GitHub-trusted CA bundles.

    JIT path: <site-packages>/silero_vad/data/silero_vad.jit  (~2.2 MiB)

USAGE
-----
    HF_HOME=/Volumes/wd_office_1/hf-cache \
        /Volumes/wd_office_1/repos/silero-vad-bench/.venv/bin/python \
        /Volumes/wd_office_1/repos/speax/benchmarks/vad/export_silero_to_onnx.py

OUTPUTS
-------
    /Volumes/wd_office_1/repos/vad/weights/silero/model.onnx
    /Volumes/wd_office_1/repos/speax/benchmarks/out/silero-onnx/onnx_parity.txt

MODEL SHAPE
-----------
Silero VAD is a stateful RNN that consumes fixed-size chunks at 16 kHz:

    - chunk size       : 512 samples (32 ms @ 16 kHz)
    - context size     : 64 samples (4 ms — last 64 of the previous chunk)
    - hidden state     : (2, B, 128) float32 (carried across chunks)

The internal `_model` (VADRNNJIT) takes the concatenated tensor
(context || chunk) of shape (B, 576). We expose this as three explicit I/O
tensors so the Go side can keep state across chunks without hidden mutation:

    INPUTS
      input    : (B, 512)        float32 — raw audio chunk, [-1, 1]
      state    : (2, B, 128)     float32 — RNN hidden state (zeros to start)
      context  : (B, 64)         float32 — last 64 samples of previous chunk
                                            (zeros for the first chunk)

    OUTPUTS
      prob     : (B, 1)          float32 — sigmoid probability of speech
      stateN   : (2, B, 128)     float32 — updated RNN hidden state
      contextN : (B, 64)         float32 — last 64 samples of `input`
                                            (feed back as `context` next call)

This matches the upstream JIT wrapper's behaviour exactly (zero diff verified
on the 10 s reference clip). For comparison, the community ONNX at
`onnx-community/silero-vad` exposes a different I/O surface:
    inputs = (input[N,T], state[2,B,128], sr[scalar int64])
    outputs = (output[N,1], stateN[2,B,128])
…with the context buffer hidden inside the graph. We deliberately surface
context as an explicit tensor so the Go service does not have to depend on
ONNX-side state mutation semantics.

GO PREPROCESSING RECIPE
-----------------------
1. Decode source to 16 kHz mono float32 in [-1, 1].
2. Maintain three pieces of state per stream:
       state    = zeros (2, 1, 128)
       context  = zeros (1, 64)
       buf      = ring buffer of incoming samples
3. Each time `buf` has >= 512 samples, slice 512 of them:
       prob, state, context = sess.Run({input: chunk, state, context})
   Drop the consumed 512 samples from the buffer. Repeat.
4. The model only operates at 16 kHz; resample upstream if needed.

PARITY
------
We compare our exported ONNX against the upstream PyTorch wrapper (same JIT
graph, ran through the same chunk loop) AND against `onnx-community/silero-vad`
on the 10 s reference clip. Both diffs are reported.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/Volumes/wd_office_1/hf-cache")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

WAV = Path("/Volumes/wd_office_1/repos/speax/benchmarks/data/ref-10s.wav")
OUT_DIR = Path("/Volumes/wd_office_1/repos/vad/weights/silero")
ONNX_PATH = OUT_DIR / "model.onnx"
REPORT_PATH = Path(
    "/Volumes/wd_office_1/repos/speax/benchmarks/out/silero-onnx/onnx_parity.txt"
)

# Upstream JIT artifact (shipped inside the `silero-vad` pip package).
import silero_vad  # noqa: E402

JIT_PATH = Path(silero_vad.__file__).parent / "data" / "silero_vad.jit"

# Reference community ONNX for parity comparison.
COMMUNITY_ONNX = (
    "/Volumes/wd_office_1/hf-cache/hub/models--onnx-community--silero-vad/"
    "snapshots/e71cae966052b992a7eca6b17738916ce0eca4ec/onnx/model.onnx"
)

# Upstream-shipped ONNX (the same one snakers4/silero-vad publishes); the
# pip package bundles it next to the .jit. We use this as the *primary*
# reference because the community spox rebuild produces different probs.
UPSTREAM_ONNX = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"

CHUNK = 512  # 32 ms @ 16 kHz
CONTEXT = 64  # samples of previous chunk fed back
STATE_SHAPE = (2, 1, 128)


class SileroVAD16k(nn.Module):
    """ONNX-friendly wrapper around the inner VADRNNJIT (16 kHz only).

    Explicit (state, context) I/O — no hidden mutation, so the Go runtime can
    manage per-stream state with plain tensor in / tensor out semantics.
    """

    __constants__ = ["context_size"]

    def __init__(self, jit_module: torch.jit.ScriptModule):
        super().__init__()
        self.inner = jit_module._model  # VADRNNJIT
        self.context_size = CONTEXT

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_full = torch.cat([context, x], dim=1)  # (B, 576)
        prob, new_state = self.inner(x_full, state)  # prob: (B, 1)
        new_context = x[:, -self.context_size :]
        return prob, new_state, new_context


def load_audio_16k_mono(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = signal.resample_poly(data, 16000, sr).astype(np.float32)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def chunk_loop_torch_wrapper(
    jit_module: torch.jit.ScriptModule, samples: np.ndarray
) -> np.ndarray:
    """Run the upstream wrapper one chunk at a time and collect probs."""
    jit_module.reset_states()
    probs = []
    for i in range(0, len(samples) - CHUNK + 1, CHUNK):
        chunk = torch.from_numpy(samples[i : i + CHUNK]).unsqueeze(0)
        with torch.no_grad():
            p = jit_module(chunk, 16000)
        probs.append(float(p.item()))
    return np.asarray(probs, dtype=np.float32)


def chunk_loop_our_onnx(sess, samples: np.ndarray) -> np.ndarray:
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    context = np.zeros((1, CONTEXT), dtype=np.float32)
    probs = []
    for i in range(0, len(samples) - CHUNK + 1, CHUNK):
        chunk = samples[i : i + CHUNK][None, :].astype(np.float32)
        prob, state, context = sess.run(
            None, {"input": chunk, "state": state, "context": context}
        )
        probs.append(float(prob[0, 0]))
    return np.asarray(probs, dtype=np.float32)


def chunk_loop_community_onnx(sess, samples: np.ndarray) -> np.ndarray:
    """Best-effort reproduction: feeds the bare 512-sample chunk + state + sr."""
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)
    probs = []
    for i in range(0, len(samples) - CHUNK + 1, CHUNK):
        chunk = samples[i : i + CHUNK][None, :].astype(np.float32)
        out, state = sess.run(None, {"input": chunk, "state": state, "sr": sr})
        probs.append(float(out[0, 0]))
    return np.asarray(probs, dtype=np.float32)


def chunk_loop_upstream_onnx(sess, samples: np.ndarray) -> np.ndarray:
    """Drive snakers4's published .onnx with manual context+state (matches JIT)."""
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)
    context = np.zeros((1, CONTEXT), dtype=np.float32)
    probs = []
    for i in range(0, len(samples) - CHUNK + 1, CHUNK):
        chunk = samples[i : i + CHUNK][None, :]
        x_full = np.concatenate([context, chunk], axis=1).astype(np.float32)
        out, state = sess.run(None, {"input": x_full, "state": state, "sr": sr})
        probs.append(float(out[0, 0]))
        context = chunk[:, -CONTEXT:]
    return np.asarray(probs, dtype=np.float32)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not JIT_PATH.exists():
        sys.exit(f"missing JIT model at {JIT_PATH}")
    print(f"loading JIT: {JIT_PATH}")
    jit_module = torch.jit.load(str(JIT_PATH))
    jit_module.eval()

    wrapper = SileroVAD16k(jit_module)
    wrapper.eval()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    x = torch.randn(1, CHUNK, dtype=torch.float32)
    state = torch.zeros(*STATE_SHAPE, dtype=torch.float32)
    context = torch.zeros(1, CONTEXT, dtype=torch.float32)

    # The inner VADRNNJIT is a torch.jit.ScriptModule that the tracer refuses
    # to descend into. Script the wrapper so the whole graph is one
    # ScriptModule, then hand that to the ONNX exporter.
    scripted = torch.jit.script(wrapper)
    print(f"exporting ONNX -> {ONNX_PATH}")
    torch.onnx.export(
        scripted,
        (x, state, context),
        str(ONNX_PATH),
        input_names=["input", "state", "context"],
        output_names=["prob", "stateN", "contextN"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch"},
            "state": {1: "batch"},
            "context": {0: "batch"},
            "prob": {0: "batch"},
            "stateN": {1: "batch"},
            "contextN": {0: "batch"},
        },
        dynamo=False,  # tracer-based exporter; RNN states behave correctly
    )

    # ------------------------------------------------------------------
    # Parity
    # ------------------------------------------------------------------
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    samples = load_audio_16k_mono(WAV)
    n_chunks = (len(samples) // CHUNK)
    samples = samples[: n_chunks * CHUNK]
    print(f"audio: {WAV.name} {len(samples)/16000:.3f}s -> {n_chunks} chunks")

    # Re-load JIT for an independent run with state reset (wrapper is mutable)
    torch_probs = chunk_loop_torch_wrapper(jit_module, samples)

    sess_ours = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    in_meta = {i.name: (i.shape, i.type) for i in sess_ours.get_inputs()}
    out_meta = {o.name: (o.shape, o.type) for o in sess_ours.get_outputs()}
    our_probs = chunk_loop_our_onnx(sess_ours, samples)

    sess_up = ort.InferenceSession(
        str(UPSTREAM_ONNX), providers=["CPUExecutionProvider"]
    )
    up_probs = chunk_loop_upstream_onnx(sess_up, samples)

    sess_comm = ort.InferenceSession(
        COMMUNITY_ONNX, providers=["CPUExecutionProvider"]
    )
    comm_probs = chunk_loop_community_onnx(sess_comm, samples)

    diff_vs_torch = float(np.max(np.abs(our_probs - torch_probs)))
    diff_vs_upstream_onnx = float(np.max(np.abs(our_probs - up_probs)))
    diff_vs_community = float(np.max(np.abs(our_probs - comm_probs)))

    # Latency: average wall time per chunk (no warmup needed, but do 3 anyway).
    for _ in range(3):
        chunk_loop_our_onnx(sess_ours, samples[:CHUNK])
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        sess_ours.run(
            None,
            {
                "input": np.zeros((1, CHUNK), dtype=np.float32),
                "state": np.zeros(STATE_SHAPE, dtype=np.float32),
                "context": np.zeros((1, CONTEXT), dtype=np.float32),
            },
        )
    chunk_us = (time.perf_counter() - t0) * 1e6 / N

    onnx_size = ONNX_PATH.stat().st_size

    lines = []
    lines.append("Silero VAD (16 kHz) -> ONNX parity report")
    lines.append(f"jit source     : {JIT_PATH}")
    lines.append(f"onnx file      : {ONNX_PATH} ({onnx_size/1024:.1f} KiB)")
    lines.append("inputs         :")
    for name, (shape, dtype) in in_meta.items():
        lines.append(f"   {name:<10s} shape={shape}  dtype={dtype}")
    lines.append("outputs        :")
    for name, (shape, dtype) in out_meta.items():
        lines.append(f"   {name:<10s} shape={shape}  dtype={dtype}")
    lines.append(f"audio          : {WAV} ({len(samples)/16000:.3f} s, {n_chunks} chunks of {CHUNK})")
    lines.append(f"diff ours vs upstream JIT     (chunk-loop)   : {diff_vs_torch:.3e}")
    lines.append(f"diff ours vs upstream .onnx   (chunk-loop)   : {diff_vs_upstream_onnx:.3e}")
    lines.append(f"diff ours vs onnx-community/silero-vad       : {diff_vs_community:.3e}")
    lines.append(
        "NOTE: the onnx-community spox rebuild diverges from the upstream JIT "
        "and the upstream-shipped .onnx (~0.9 max-abs). Our export matches the "
        "upstream JIT + upstream .onnx; the community ONNX is *not* a reliable "
        "reference, so we do not gate parity on it."
    )
    lines.append(
        f"parity (<1e-3 vs upstream JIT + upstream onnx) : "
        f"{'PASS' if diff_vs_torch < 1e-3 and diff_vs_upstream_onnx < 1e-3 else 'FAIL'}"
    )
    lines.append(f"onnxruntime    : {ort.__version__} (CPUExecutionProvider)")
    lines.append(f"per-chunk lat  : {chunk_us:.1f} us/chunk over {N} runs")
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    print()
    print(text)
    print(f"report written -> {REPORT_PATH}")

    ok = diff_vs_torch < 1e-3 and diff_vs_upstream_onnx < 1e-3
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
