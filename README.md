# vspy

A Python port of [VoiceSauce](http://www.phonetics.ucla.edu/voicesauce/), a MATLAB-based tool for acoustic voice quality analysis. `vspy` extracts per-frame acoustic features from `.wav` files and writes them to a single combined CSV.

## What it does

Given a folder of `.wav` files (or a single file), `vspy` computes a set of acoustic features at 1 ms intervals and returns a pandas DataFrame — one row per frame, one column per feature.

**Extracted features:**

| Feature | Description |
|---|---|
| `f0_praat` | Fundamental frequency (F0) via Praat / parselmouth |
| `f0_snack` | F0 via Snack (LPC-based pitch tracker) |
| `F1–F3_praat` | Formant frequencies via Praat |
| `F1–F3_snack` | Formant frequencies via Snack LPC |
| `H1, H2, H4` | Amplitudes of the 1st, 2nd, and 4th harmonics |
| `A1, A2, A3` | Harmonic amplitudes nearest F1, F2, F3 |
| `H2K, F2K, H5K` | Energy in the 2 kHz and 5 kHz bands |
| Spectral tilts | H1–H2, H1–H4, H1–A1, H1–A2, H1–A3, H2K–H5K, and their corrected variants |

## Installation

```bash
pip install -e .
```

Requires Python 3.9+. Dependencies: `numpy`, `pandas`, `soundfile`, `praat-parselmouth`, `librosa`, `pydub`.

## Usage

```python
from vspy import vspy

# Run on a folder of .wav files
df = vspy("path/to/wavs/")

# Run on a single file
df = vspy("audio.wav")

# Optional: filter to labeled segments using a TextGrid
df = vspy("path/to/wavs/", textgrid_dir="path/to/textgrids/", tier="vowel")

# Optional: select specific features or change frameshift
df = vspy("path/to/wavs/", features=["f0_praat", "formants_praat"], frameshift_ms=1)
```

Output is written to `output.csv` by default and also returned as a DataFrame. Columns: `t_ms` (left-edge frame timestamp in milliseconds), one column per feature, and `filename`.

## Frame grid

- Hop size: 1 ms (`frameshift_ms=1`)
- Window length: 25 ms
- `t_ms` is a left-edge integer millisecond timestamp (frame 0 = 0 ms, frame 1 = 1 ms, …)

## Project status

This is a work-in-progress port targeting feature parity with VoiceSauce. Core F0 and formant extraction (Praat and Snack) plus the full harmonic/spectral-tilt family are implemented. STRAIGHT and SHR pitch methods are not yet supported.
