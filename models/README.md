# `models/` — source-tensor → ONNX conversion + license tracking

This directory holds the **reproducible** side of the per-backend ONNX
weights that ship in `weights/<backend>/model.onnx`. Three jobs:

1. **Conversion scripts** (`models/<backend>/convert.py`) that take the
   upstream PyTorch / NeMo / JIT checkpoint and re-emit the exact
   `model.onnx` byte we ship.
2. **License + attribution tracking** (`models/LICENSES/`,
   `models/<backend>/NOTICE.md`) — including the more involved
   redistribution requirements like NVIDIA's NOML.
3. **Validation** (`models/validate.py`) — re-runs each conversion in a
   tempdir and asserts the output is byte-identical (or numerically
   equivalent within tolerance) to the committed reference. All 4
   convert.py-backed models pass byte-identical on 2026-05-30.

**Important**: the existing committed `weights/<backend>/model.onnx`
files are the **reference**. Do not delete them. The convert scripts
are graded against them. Replacing the reference requires a deliberate
"rebaseline" — re-run all five convert.py's, smoke-test the binary,
update the bench parity reports.

## Layout

```
models/
├── README.md                        this file
├── validate.py                      byte-identical assertion harness
├── LICENSES/
│   ├── apache-2.0.txt               canonical Apache-2.0 (fsmn + firered)
│   ├── pyannote.LICENSE             MIT (pyannote)
│   ├── silero.LICENSE               MIT (silero)
│   ├── fsmn-vad.LICENSE             pointer to apache-2.0
│   ├── firered-vad.LICENSE          pointer to apache-2.0
│   └── marblenet.LICENSE            NVIDIA Open Model License
├── pyannote/
│   └── NOTICE.md                    (no convert.py — we use the
│                                     onnx-community port verbatim;
│                                     see NOTICE.md for rationale)
├── fsmn-vad/
│   ├── convert.py                   PyTorch → ONNX
│   └── NOTICE.md
├── firered-vad/
│   ├── convert.py                   PyTorch → ONNX
│   └── NOTICE.md
├── silero/
│   ├── convert.py                   JIT → ONNX
│   └── NOTICE.md
└── marblenet/
    ├── convert.py                   NeMo (.nemo) → ONNX
    └── NOTICE.md
```

## Licensing matrix

| Backend | License | Permissive? | Special handling |
|---|---|---|---|
| pyannote   | MIT                          | ✓ | preserve copyright |
| fsmn-vad   | Apache-2.0                   | ✓ | standard attribution |
| firered-vad| Apache-2.0                   | ✓ | standard attribution |
| silero     | MIT                          | ✓ | preserve copyright |
| marblenet  | NVIDIA Open Model License    | ✓ (commercial-ok with strings) | must redistribute license + attribute NVIDIA |

4 of 5 are fully permissive (MIT or Apache-2.0). NVIDIA's NOML is
permissive for commercial use + derivative works, but requires
distribution of the license text and attribution. See
[`marblenet/NOTICE.md`](marblenet/NOTICE.md) for the full attribution
text and the redistribution checklist.

## Validation

`models/validate.py` runs each `convert.py` into a tempdir and asserts
the output `.onnx` matches the checked-in `weights/<backend>/model.onnx`
byte-for-byte. All 4 backends with conversion scripts (silero, fsmn-vad,
firered-vad, marblenet) were verified byte-identical on 2026-05-30
against their respective `<engine>-bench` venvs (parity vs PyTorch /
NeMo at noise floor: ~2e-6 max-abs).

Each convert.py expects to find a Python venv with the upstream package
(torch + funasr / nemo_toolkit[asr] / silero_vad / etc.). Point
validate.py at them via per-backend env vars:

```bash
SILERO_VENV=/path/to/silero-vad-bench/.venv \
FSMN_VENV=/path/to/fsmn-vad-bench/.venv \
FIRERED_VENV=/path/to/firered-vad-bench/.venv \
MARBLENET_VENV=/path/to/marblenet-bench/.venv \
python models/validate.py
```

Backends whose env var isn't set are reported as "missing venv" and
counted as a failure — set just the env vars for whichever backend
you have the deps for. pyannote is always SKIP (no in-repo convert.py;
we use the onnx-community port verbatim).

The validator falls back to numerical-equivalence comparison
(`max_abs_tol`) if a backend is marked `byte_identical: False` —
useful if a future upstream toolchain change introduces nondeterminism
without invalidating the model. Today all 4 are `byte_identical: True`.

Wired into `release.sh` between `04 tests` and `05 build_bin` (TODO —
gated on venv detection so the wire-up skips cleanly on hosts that
don't have the upstream venvs installed).

The bench parity reports at
`speax/benchmarks/out/<backend>-onnx/onnx_parity.txt` are written as a
side effect of each convert.py run — they record the max-abs / mean-abs
diff between the upstream PyTorch / NeMo forward and the exported ONNX,
and stay around as audit history.

## How to re-export a model from source

```bash
. /path/to/silero-vad-bench/.venv/bin/activate
python models/silero/convert.py
# Output: weights/silero/model.onnx (overwrites the reference; revert
#         the diff unless you intend to rebaseline)
```

## Embedding license info into the binary (TODO)

Planned: `bin/vad --licenses` subcommand that dumps every third-party
model's license text + attribution notice (this directory's
`LICENSES/*` and `<backend>/NOTICE.md` files, embedded via
`go:embed`). This satisfies the NOML redistribution requirement in a
self-contained way — any operator running our binary can dump the
attribution chain without consulting external files.

Implementation hook: a new `internal/licenses/` Go package with
`//go:embed LICENSES/* */NOTICE.md` (rooted at `models/`), exposed via
a `LicensesText()` function `cmd/vad/main.go` wires to the new flag.
Not yet implemented; tracked in the repo's TODO list.
