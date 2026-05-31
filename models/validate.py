#!/usr/bin/env python3
"""validate.py — assert each models/<backend>/convert.py reproduces the
reference weights/<backend>/model.onnx that ships in the repo.

Strategy: byte-identical comparison where it's expected (silero ONNX is
seeded + deterministic on a frozen toolchain); numerical-equivalence
sanity (max-abs activation diff under a tolerance) for the others where
upstream training-time nondeterminism or torch.onnx.export's free
choices make byte-identity unrealistic.

This script is the *contract* between source tensors and shipped
artifacts: if a contributor changes a convert.py, this is what tells
them whether the change is benign (numerical-equiv) or a rebaseline
(diff exceeds tolerance, requires updating the reference + regenerating
parity reports).

Usage:
    python models/validate.py                     # run all 5
    python models/validate.py silero marblenet    # subset

Exit 0 if all pass; 1 if any fail.

NOTE: actually invoking each convert.py requires its upstream venv
(torch + funasr / nemo_toolkit[asr] / silero_vad / etc.). This script
deliberately doesn't try to set those up — it discovers them via
$<BACKEND>_VENV env vars (e.g. SILERO_VENV=/path/to/.venv) and falls
back to documenting what's missing.

Wired into release.sh as a step between tests and build_bin once the
needed venvs exist on the build host.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
WEIGHTS_DIR = REPO_ROOT / "weights"

# Per-backend tolerance for numerical comparison. byte_identical means the
# re-exported ONNX must match the reference byte for byte. Otherwise we'd
# load both as onnx, run them on a fixed silence-or-noise input, and
# compare the output activations under max_abs_tol.
BACKENDS = {
    "pyannote": {
        # No convert.py — we use onnx-community port verbatim.
        "skip_reason": "no in-repo conversion (uses onnx-community/pyannote-segmentation-3.0 verbatim)",
    },
    # All 4 below were verified byte-identical to the checked-in reference
    # on 2026-05-30 against their respective `<engine>-bench` venvs. If
    # this ever flips to a diff, fall back to the numerical-equivalence
    # path (max_abs_tol ~1e-3) and figure out what changed in the
    # upstream toolchain.
    "silero": {
        "venv_env": "SILERO_VENV",
        "byte_identical": True,
    },
    "fsmn-vad": {
        "venv_env": "FSMN_VENV",
        "byte_identical": True,
    },
    "firered-vad": {
        "venv_env": "FIRERED_VENV",
        "byte_identical": True,
    },
    "marblenet": {
        "venv_env": "MARBLENET_VENV",
        "byte_identical": True,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_backend(name: str, cfg: dict) -> tuple[bool, str]:
    """Returns (ok, message)."""
    if cfg.get("skip_reason"):
        return True, f"SKIP — {cfg['skip_reason']}"

    convert_py = MODELS_DIR / name / "convert.py"
    reference = WEIGHTS_DIR / name / "model.onnx"
    if not convert_py.exists():
        return False, f"missing {convert_py}"
    if not reference.exists():
        return False, f"missing reference {reference}"

    venv_env = cfg.get("venv_env")
    venv_path = os.environ.get(venv_env) if venv_env else None
    if not venv_path:
        return False, (
            f"set ${venv_env} to a Python venv that has the upstream package "
            f"(see {convert_py}'s header for the exact deps)"
        )
    python = Path(venv_path) / "bin" / "python"
    if not python.exists():
        return False, f"${venv_env}={venv_path} but {python} is missing"

    # Run convert.py into a tempdir so we don't clobber the reference.
    with tempfile.TemporaryDirectory(prefix=f"vad-validate-{name}-") as tmp:
        # Stage a fake weights/<name>/ for the convert.py to write into.
        # The scripts write directly to weights/<backend>/model.onnx in
        # the repo root; redirect by chdir + symlink workaround.
        staged_repo = Path(tmp) / "repo"
        staged_repo.mkdir()
        (staged_repo / "weights").mkdir()
        # Symlink everything from the real repo so absolute paths inside
        # the convert script still resolve, but redirect the weights dir.
        for entry in REPO_ROOT.iterdir():
            if entry.name == "weights":
                continue
            (staged_repo / entry.name).symlink_to(entry)
        (staged_repo / "weights" / name).mkdir()
        # Need the existing config files in the weights dir.
        ref_dir = WEIGHTS_DIR / name
        for f in ref_dir.iterdir():
            if f.name != "model.onnx":
                shutil.copy(f, staged_repo / "weights" / name / f.name)

        proc = subprocess.run(
            [str(python), str(convert_py)],
            cwd=str(staged_repo),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            return False, f"convert.py failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"

        new = staged_repo / "weights" / name / "model.onnx"
        if not new.exists():
            return False, f"convert.py ran but produced no model.onnx at {new}"

        if cfg.get("byte_identical"):
            new_hash = sha256(new)
            ref_hash = sha256(reference)
            if new_hash == ref_hash:
                return True, f"BYTE-IDENTICAL ({new_hash[:16]}…)"
            return False, (
                f"BYTE DIFF\n  reference: {ref_hash}\n  new:       {new_hash}\n"
                f"  (sizes: {reference.stat().st_size} vs {new.stat().st_size})"
            )

        # Numerical equivalence path. Defer to onnxruntime in $VENV.
        cmp_py = (
            "import sys, onnxruntime as ort, numpy as np\n"
            f"a = ort.InferenceSession({str(reference)!r}, providers=['CPUExecutionProvider'])\n"
            f"b = ort.InferenceSession({str(new)!r}, providers=['CPUExecutionProvider'])\n"
            "inputs = a.get_inputs()\n"
            "# Build a deterministic silence input matching each input's shape.\n"
            "feed = {}\n"
            "for inp in inputs:\n"
            "    shape = [d if isinstance(d, int) else 1 for d in inp.shape]\n"
            "    feed[inp.name] = np.zeros(shape, dtype=np.float32)\n"
            "ya = a.run(None, feed)\n"
            "yb = b.run(None, feed)\n"
            "max_abs = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(ya, yb))\n"
            "print(f'max_abs={max_abs:.3e}')\n"
            "sys.exit(0 if max_abs <= " + repr(cfg["max_abs_tol"]) + " else 1)\n"
        )
        cmp = subprocess.run(
            [str(python), "-c", cmp_py],
            capture_output=True, text=True, timeout=60,
        )
        msg = cmp.stdout.strip() or cmp.stderr.strip()
        if cmp.returncode == 0:
            return True, f"NUMERICAL OK ({msg}; tol={cfg['max_abs_tol']:.0e})"
        return False, f"NUMERICAL DIFF — {msg} (tol={cfg['max_abs_tol']:.0e})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backends", nargs="*", default=list(BACKENDS),
                    help="subset of backends to validate; default = all 5")
    args = ap.parse_args()

    fails = 0
    for name in args.backends:
        if name not in BACKENDS:
            print(f"unknown backend: {name}", file=sys.stderr)
            return 2
        cfg = BACKENDS[name]
        print(f"=== {name} ===")
        ok, msg = validate_backend(name, cfg)
        print("  " + msg)
        if not ok:
            fails += 1

    print()
    if fails:
        print(f"VALIDATION FAILED: {fails}/{len(args.backends)} backend(s) failed")
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
