# vspy
VoiceSauce Python (VSPY) is a Python implementation of UCLA's
[VoiceSauce](https://www.phonetics.ucla.edu/voicesauce/). It also includes
nasality measurements adapted from Dr. Will Styler's [Nasality AutoMeasurement
Praat script](https://github.com/stylerw/styler_praat_scripts/blob/master/nasality_automeasure/NasalityAutomeasure.praat)
and spectral-moment measurements adapted from Dr. Christian DiCanio's [spectral
moments Praat script](https://www.acsu.buffalo.edu/~cdicanio/scripts/Time_averaging_for_fricatives_4.0.praat).

## Features

The default `vspy()` run produces:

- Pitch: `f0_straight`, `f0_praat`, `f0_snack`, `f0` (SHR pitch), and `shr`.
- Praat formants and bandwidths: `F1_praat` through `F4_praat` and `B1_praat` through `B4_praat`.
- Snack formants and bandwidths: `F1_snack` through `F4_snack` and `B1_snack` through `B4_snack`.
- Harmonic amplitudes: `H1`, `H2`, and `H4`.
- Formant-region amplitudes: `A1`, `A2`, and `A3`.
- Spectral-band measurements: `H2K`, `F2K`, and `H5K`.
- Formant-corrected amplitudes: `H1c`, `H2c`, `H4c`, `A1c`, `A2c`, `A3c`, and `H2Kc`.
- Spectral tilts: `H1H2`, `H2H4`, `H1A1`, `H1A2`, `H1A3`, `H4H2K`, and `H2KH5K`.
- Cepstral peak prominence: `CPP`.
- Harmonics-to-noise ratios: `HNR05`, `HNR15`, `HNR25`, and `HNR35`.
- Energy: `Energy`.
- Strength of excitation: `epoch` and `SoE`.
- Spectral moments: `spectral_cog`, `spectral_sdev`, `spectral_skew`, and `spectral_kurt`.
- Nasality: `amp_p0`, `freq_p0`, `p0_id`, `p0prominence`, `a1p0_h1`, `a1p0_h2`, `a1p0_h3`, `a1p0`, `a1p0_compensated`, `freq_p1`, `amp_p1`, `a1p1`, `a1p1_Compensated`, `a3p0`, and `vwl_duration`.

The default hop is 1 ms. Spectral-moment and nasality windows are 25 ms for hops up to 25 ms. For larger hops, their windows expand to the hop length so there are no gaps between adjacent windows.

## Installation

Python 3.9 or newer is required.

```bash
pip install -e .
```

Dependencies are NumPy, SciPy, pandas, PyArrow, SoundFile, praat-parselmouth,
librosa, and pydub.

## Sample code

Import package
```python
from vspy import vspy
```
Run on a single audio file
```python
df = vspy(
    # every vspy call needs a path to where your WAV files are
    source = "/path/to/audio.wav",
    # and a path to where the output should be saved
    output_file="/path/to/results/voice_features.csv",
)
```

Run a folder of WAV files:
```python
df = vspy(
    source = "/path/to/wavs/",
    output_file="/path/to/results/voice_features.csv",
)
```

Run on a folder of WAV files and corresponding textgrids
```python
df = vspy(
    source="/path/to/audio_files/",
    # textgrid directory
    textgrid_dir="/path/to/textgrid_files/",
    # one tier name, or a list of tier names
    tier=["word", "vowel"],
    output_file="/path/to/results/voice_features.csv",
)
```

Really big files can also be saved as a parquet 
```python
df = vspy(
    source="/path/to/audio_files/",
    output_file="/path/to/results/voice_features.parquet",
)
```

### Tunable parameters

- `features=None`: run every registered extractor. Supply a list of extractor
  names to run a subset.
- `frameshift_ms=1`: time in milliseconds between analysis frames. TextGrid
  labels follow each frame's `t_ms`: interval starts are included and ends
  are excluded.
- `textgrid_dir=None`: directory containing matching `.TextGrid` files.
- `tier=None`: TextGrid tier used to label and select intervals. Pass one tier
  name or a list such as `["word", "vowel"]`. Each selected tier gets its own
  output column, and frames labeled on any selected tier are retained.
- `f0_source="straight"`: pitch track used by dependent measurements. Available
  sources are `"straight"`, `"praat"`, and `"snack"`.
- `formant_source="praat"`: formant track used by dependent measurements.
  Available sources are `"praat"` and `"snack"`.
- `nasality_p1search=950.0`: center frequency in Hz of the P1 search region.
- `nasality_tpnum=5`: number of nasality measurement points per file or
  selected TextGrid interval.
- `nasality_scoot=0.0`: distance in seconds that the first and last nasality
  measurement points move inward from interval boundaries.
