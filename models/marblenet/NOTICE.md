# NVIDIA Frame VAD Multilingual MarbleNet v2.0 — attribution + provenance

**License**: NVIDIA Open Model License — see
[`../LICENSES/marblenet.LICENSE`](../LICENSES/marblenet.LICENSE).
**Canonical license URL**:
https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license

**Source repo (NeMo)**: https://github.com/NVIDIA/NeMo
**Source model**: https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0
**Upstream org**: NVIDIA NeMo Team
**Site**: https://developer.nvidia.com/nemo-framework

## What's bundled in this repo

```
weights/marblenet/
  model.onnx          362 KB    exported by this repo's convert.py from
                                frame_vad_multilingual_marblenet_v2.0.nemo
  preprocessor.yaml             NeMo preprocessor config (used by
                                melspec/ to reproduce the feature pipeline)
  url.txt                       CDN-style URL for the Fetch RPC
```

## Attribution (required by license)

This product includes the **NVIDIA Frame VAD Multilingual MarbleNet
v2.0** model, developed by NVIDIA Corporation, used under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license).

When redistributing the `vad` binary or any derivative that includes
this model — embedded or otherwise — the following must accompany the
distribution:

1. A copy of (or a link to) the NVIDIA Open Model License. We satisfy
   this by shipping `models/LICENSES/marblenet.LICENSE` in the source
   tree, surfacing the same content via the planned `bin/vad --licenses`
   subcommand, and including it in the GitHub Release zip.

2. The above attribution sentence (or equivalent).

3. The original model card link
   (https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)
   reachable from project documentation. We surface it in the
   [README backends table](../../README.md) and at runtime via the
   `Fetch` RPC's `url.txt` indirection.

4. Compliance with the Acceptable Use restrictions in the license
   (no misinformation, illegal activity, etc. — see the canonical
   license URL for the current list).

## Conversion

`models/marblenet/convert.py` regenerates `weights/marblenet/model.onnx`
from the upstream `.nemo` checkpoint. Parity vs the NeMo PyTorch
forward is `max_abs ~ 2e-6` per the report at
`speax/benchmarks/out/marblenet-onnx/onnx_parity.txt`.

DO NOT delete the checked-in `weights/marblenet/model.onnx`. The
release pipeline's validation step will assert that a re-export from
the current convert.py matches the reference.

## What we did NOT change

No modifications to the upstream model architecture, weights, or
preprocessor config. The ONNX export wraps the NeMo `EncDecClassifier`
forward via the standard NeMo export path, and `melspec/` reproduces
the preprocessor in pure Go to feed the exported encoder its expected
log-mel features.
