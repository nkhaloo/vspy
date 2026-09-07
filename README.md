# vspy

A Python port of [VoiceSauce](http://www.phonetics.ucla.edu/voicesauce/), a MATLAB-based tool for acoustic voice quality analysis. `vspy` extracts per-frame acoustic features from `.wav` files and writes them to a single combined CSV.

## What it does

Given a folder of `.wav` files (or a single file), `vspy` computes a set of acoustic features at 1 ms intervals and returns a pandas DataFrame — one row per frame, one column per feature.

**Extracted features:**

| Feature | Description |
|---|---|
| `f0_straight` | VoiceSauce TANDEM-STRAIGHT F0, the default for dependent features |
| `f0_praat` | Fundamental frequency (F0) via Praat / parselmouth |
| `f0_snack` | F0 via Snack (LPC-based pitch tracker) |
| `F1–F3_praat` | Formant frequencies via Praat |
| `F1–F3_snack` | Formant frequencies via Snack LPC |
| `H1, H2, H4` | Amplitudes of the 1st, 2nd, and 4th harmonics |
| `A1, A2, A3` | Harmonic amplitudes nearest F1, F2, F3 |
| `H2K, F2K, H5K` | Energy in the 2 kHz and 5 kHz bands |
| Spectral tilts | H1–H2, H1–H4, H1–A1, H1–A2, H1–A3, H2K–H5K, and their corrected variants |
| `spectral_moments` | `spectral_cog`, `spectral_sdev` (Hz), `spectral_skew`, and `spectral_kurt` (excess kurtosis) |

## Installation

```bash
pip install -e .
```

Requires Python 3.9+. Dependencies: `numpy`, `scipy`, `pandas`, `soundfile`, `praat-parselmouth`, `librosa`, `pydub`.

## Usage

```python
from vspy import vspy

# Run on a folder of .wav files
df = vspy("path/to/wavs/")

# Run on a single file
df = vspy("audio.wav")

# VoiceSauce defaults for both pitch and formants
df = vspy("audio.wav", f0_source="straight", formant_source="snack")

# Optional: filter to labeled segments using a TextGrid
df = vspy("path/to/wavs/", textgrid_dir="path/to/textgrids/", tier="vowel")

# Optional: select specific features or change frameshift
df = vspy("path/to/wavs/", features=["f0_praat", "formants_praat"], frameshift_ms=1)

```

Spectral moments are included by default, or can be selected with
`features=["spectral_moments"]`. They use the configured `frameshift_ms`
(1 ms by default), with a 15 ms Hann window centered at each frame time.
Power-weighted moments use bins from 300 Hz through the original recording's
Nyquist frequency; this feature bypasses the voice pipeline's 16 kHz resampling.
Incomplete edge windows and silent windows return NaN. No F0 mask or output
smoothing is applied to these columns. These definitions follow Christian
DiCanio's `spectral moments` Praat script, but calculate one spectrum per frame
instead of averaging six spectra across a labeled interval. Values therefore
are not directly equivalent to the original interval-level measurements.

To compare against the original Praat script on explicit `vowel`/`vowels`
TextGrid tiers in the test corpus and sanity recordings, run
`python -m test.compare_spectral_moments`. The runner saves raw Praat results,
vspy frame values, full-segment and middle-80% averages, and error metrics in
`test/spectral_moments_vowel_results/`. Add `--sanity-only` for the three
sanity recordings, using a separate `--outdir` to keep runs distinct.

Nasality measurements use a separate pipeline:

```python
from vspy import vspy_nasality

df = vspy_nasality("audio.wav", p1search=950, tpnum=5, scoot=0.0)
```

Output is written to `output.csv` by default and also returned as a DataFrame. Columns: `t_ms` (left-edge frame timestamp in milliseconds), one column per feature, and `filename`.

## Frame grid

- Hop size: 1 ms (`frameshift_ms=1`)
- Window length: 25 ms
- `t_ms` is a left-edge integer millisecond timestamp (frame 0 = 0 ms, frame 1 = 1 ms, …)

## Project status

This is a work-in-progress port targeting feature parity with VoiceSauce. TANDEM-STRAIGHT is the default F0 source for harmonic, spectral-tilt and voice-quality calculations. Praat, Snack and SHR remain available as comparison outputs. See [STRAIGHT implementation and evaluation](docs/straight_port.md) for compatibility details and reproduction commands.

The restored corpus's [full evaluation report](test/straight_results/report.md) includes correlations and errors for every comparable feature.
