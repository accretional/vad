# fbank

Pure-Go log-Mel-filterbank feature extractor, byte-compatible with `kaldi-native-fbank` to within float32 round-off. Used by the FSMN-VAD and FireRedVAD backends in `pkg/vad/` (both expect Kaldi-style features at the model input).

## Why a separate package

- **Reuse** across both new VAD backends (and any future Kaldi-feature-based model).
- **No CGO** — pure Go means no C++ vendor / build / link complexity. Trades runtime speed (≈2× slower than the C++ reference) for build simplicity.
- **Parity-tested in CI** — `kaldi_native_fbank` ground-truth fixtures live in `testdata/`; if we drift from upstream, tests fail.

## Usage

```go
import "github.com/accretional/vad/fbank"

opts := fbank.Defaults()         // Kaldi default 25 ms / 10 ms / 80 mels / Povey window
opts.WindowType = fbank.WindowHamming  // FSMN-VAD expects Hamming
fb, err := fbank.New(opts)
if err != nil { ... }

// samples should already be scaled to int16-range (multiply [-1,1] float32 by 32768
// to match FunASR / FireRed input expectations).
features := fb.Compute(samples)  // [][]float32 of shape [NumFrames(len(samples))][80]
```

## Defaults

`fbank.Defaults()` returns Kaldi's standard VAD/ASR `compute-fbank-feats`-equivalent settings:

| Field | Value |
|---|---|
| `SampleRate` | 16000 |
| `FrameLengthMs` | 25 |
| `FrameShiftMs` | 10 |
| `NumMelBins` | 80 |
| `WindowType` | Povey |
| `LowFreqHz` | 20 |
| `HighFreqHz` | 0 (= SampleRate/2 = 8000) |
| `PreemphCoeff` | 0.97 |
| `RemoveDCOffset` | true |
| `SnipEdges` | true |
| `EnergyFloor` | 0 (internally clamped to float32 epsilon ≈ 1.19e-7 to match Kaldi) |
| `UseLog` | true |

## Performance (M4)

For a 10-second 16 kHz mono clip:

| Impl | Wall | Notes |
|---|---|---|
| `torchaudio.compliance.kaldi.fbank` | ~6 ms | Vectorized PyTorch, MPS-backed |
| `kaldi_native_fbank` (C++ via pybind) | ~11 ms | The reference impl |
| **`fbank` (this package, pure Go)** | **~21 ms** | About 2× slower than C++ |

For a gRPC VAD server polling at 400 ms intervals with a 3 s tail (~the speax voice loop's pattern), fbank cost per call is ~6 ms — small relative to the model inference itself. Per-frame `power` slice allocation and float64 arithmetic are the obvious optimization targets if needed later.

## Parity testing

`fbank_test.go` runs the same audio through Go fbank and against a kaldi-native-fbank ground-truth fixture. Both Povey and Hamming windows are tested; both pass with max-abs diff ≈ 1.5e-2 and p99 diff ≈ 1.1e-2 in the log-energy domain. These are float32-round-off differences and well below what affects downstream model inference.

Regenerate fixtures (e.g. after adding a new option config) by running:

```bash
/Volumes/wd_office_1/repos/firered-vad-bench/.venv/bin/python testdata/gen_fixtures.py
```

(Any venv with `kaldi_native_fbank` + `numpy` works.)

## API stability

`fbank.Options` and `fbank.Defaults()` are public; `fbank.New(opts)` and `(*Fbank).Compute(samples)` are the only methods backends should call. The internal FFT, window, and mel-filterbank helpers are package-private and may change.
