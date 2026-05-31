# FireRed DFSMN-VAD — attribution + provenance

**License**: Apache License, Version 2.0 — see
[`../LICENSES/apache-2.0.txt`](../LICENSES/apache-2.0.txt) and
[`../LICENSES/firered-vad.LICENSE`](../LICENSES/firered-vad.LICENSE).

**Source repo (PyTorch)**: https://github.com/FireRedTeam/FireRedASR
**Source model**: https://huggingface.co/FireRedTeam/FireRedVAD
**Upstream org**: FireRedTeam (RedNote / Xiaohongshu ASR team)
**Site**: https://fireredteam.github.io/

## What's bundled in this repo

```
weights/firered-vad/
  model.onnx           2.34 MB    exported by this repo's convert.py
  config.yaml                     upstream FireRedVAD config (unmodified)
  cmvn.ark                        upstream CMVN stats (unmodified)
  cmvn_means.f32                  CMVN means in f32 form (derived; see convert.py)
  cmvn_istd.f32                   CMVN inverse-std in f32 form (derived)
  url.txt                         CDN-style URL for the Fetch RPC
```

## Conversion

`models/firered-vad/convert.py` regenerates `weights/firered-vad/model.onnx`
from the upstream PyTorch checkpoint. Parity vs the original PyTorch
forward pass is `max_abs ~ 3e-7` per the report at
`speax/benchmarks/out/firered-vad/onnx_parity.txt`.

DO NOT delete the checked-in `weights/firered-vad/model.onnx`. The release
pipeline's validation step will assert that a re-export from the current
convert.py produces a numerically-equivalent file. The checked-in copy
is the reference.

## Attribution

FireRed DFSMN-VAD by FireRedTeam. Used under Apache License 2.0. The
ONNX export wraps the upstream non-streaming variant; the upstream
streaming variant (Stream-VAD) is a separate model we have not
ported.
