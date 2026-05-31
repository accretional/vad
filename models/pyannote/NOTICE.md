# Pyannote Segmentation 3.0 — attribution + provenance

**License**: MIT — see [`../LICENSES/pyannote.LICENSE`](../LICENSES/pyannote.LICENSE).

**Source repo (PyTorch)**: https://github.com/pyannote/pyannote-audio
**Source model**: https://huggingface.co/pyannote/segmentation-3.0
**ONNX port we use**: https://huggingface.co/onnx-community/pyannote-segmentation-3.0
**Site**: https://www.pyannote.ai/

## What's bundled in this repo

`weights/pyannote/model.onnx` is a verbatim copy of the `onnx/model.onnx`
file from the onnx-community port linked above. We did not re-export
from PyTorch — the onnx-community port is the canonical ONNX
distribution for this model (transformers.js compatible).

`weights/pyannote/url.txt` points the `Fetch` RPC at the GitHub raw URL
of the committed `model.onnx`, so browser clients can pull the bytes
directly without proxying through our gRPC server.

## Conversion

This backend does NOT have a `convert.py` here because there's no
PyTorch → ONNX step we own. The conversion was done by the onnx-
community port maintainers. If the source `pyannote/segmentation-3.0`
model is updated in a way that needs re-exporting, the right next step
is filing an issue against `onnx-community/pyannote-segmentation-3.0`
or contributing the re-export there, rather than adding a duplicate
exporter here.

## Attribution

Pyannote Segmentation 3.0 by Hervé Bredin et al. Used under MIT license.
The onnx-community port is also MIT.
