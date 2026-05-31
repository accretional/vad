# Silero VAD — attribution + provenance

**License**: MIT — see [`../LICENSES/silero.LICENSE`](../LICENSES/silero.LICENSE).

**Source repo (PyTorch JIT)**: https://github.com/snakers4/silero-vad
**Upstream org**: Silero Team
**Site**: https://silero.ai/

## What's bundled in this repo

```
weights/silero/
  model.onnx      1.26 MB     exported by this repo's convert.py from the
                              upstream silero_vad.jit checkpoint
  url.txt                     CDN-style URL for the Fetch RPC
```

## Conversion

`models/silero/convert.py` regenerates `weights/silero/model.onnx` from
the upstream JIT-traced checkpoint shipped via the `silero_vad` pip
package.

Parity vs the upstream JIT forward (and vs the upstream pre-built ONNX
that ships in the pip package) is bit-identical for ONNX and `max_abs
~ 1.2e-6` vs JIT, per the report at
`speax/benchmarks/out/silero-onnx/onnx_parity.txt`. Note: the
`onnx-community/silero-vad` HF rebuild diverges ~0.9 max-abs from the
upstream — our export tracks the upstream JIT + upstream ONNX, not
the community rebuild.

DO NOT delete the checked-in `weights/silero/model.onnx`. The release
pipeline's validation step will assert that a re-export from the
current convert.py matches the reference.

## Attribution

Silero VAD by Silero Team. Used under MIT license.
