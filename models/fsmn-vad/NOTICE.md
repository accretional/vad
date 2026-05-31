# FunASR FSMN-VAD — attribution + provenance

**License**: Apache License, Version 2.0 — see
[`../LICENSES/apache-2.0.txt`](../LICENSES/apache-2.0.txt) and
[`../LICENSES/fsmn-vad.LICENSE`](../LICENSES/fsmn-vad.LICENSE).

**Source repo (PyTorch)**: https://github.com/modelscope/FunASR
**Source model**: https://huggingface.co/funasr/fsmn-vad
**Upstream org**: Alibaba DAMO Academy / FunASR
**Site**: https://www.funasr.com/

## What's bundled in this repo

```
weights/fsmn-vad/
  model.onnx       1.73 MB    exported by this repo's convert.py from the
                              upstream PyTorch checkpoint at funasr/fsmn-vad
  config.yaml                 upstream FunASR config (unmodified)
  am.mvn                      upstream CMVN stats (unmodified)
  url.txt                     CDN-style URL for the Fetch RPC
```

## Conversion

`models/fsmn-vad/convert.py` regenerates `weights/fsmn-vad/model.onnx`
from the upstream PyTorch checkpoint. Parity vs the original PyTorch
forward pass is `max_abs ~ 2e-6` per the report at
`speax/benchmarks/out/fsmn-vad/onnx_parity.txt`.

DO NOT delete the checked-in `weights/fsmn-vad/model.onnx`. The release
pipeline's validation step (`models/validate.py`, planned) will assert
that a re-export from the current convert.py produces a byte-identical
or numerically-equivalent file. The checked-in copy is the reference.

## Attribution

FSMN-VAD by Alibaba DAMO Academy, distributed via FunASR. Used under
Apache License 2.0. No code changes were made to the upstream model
architecture; the ONNX export wraps it with the standard
torch.onnx.export plumbing.
