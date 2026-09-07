# Spectral moments comparison

Compared 271 /s/ intervals from 264 audio files in test/wav,
using matching test/textgrid annotations. Every interval labeled exactly `s`
on an interval tier was selected, matching the original script's default label.
The original Praat script was executed unchanged through Parselmouth; its form
arguments use the appropriate tier and original sample rate (no upsampling).
Defaults retained: six 15 ms windows, 300 Hz cutoff, middle 80% of the segment.

vspy's registered spectral_moments extractor ran at a 1 ms hop on the original
audio. Arithmetic means of valid frame moments were calculated separately for
frame centers in [start, end) and in the middle 80% of each interval. NaNs are
excluded; frame counts are in segment_comparison.csv. Frames are computed on
the continuous recording, so windows near segment boundaries may cross them.

0 intervals violate the original
script's recommendation that six window lengths total no more than 1.6 times
the segment duration; these are flagged and retained. 7 read/run
errors are recorded in errors.json.

## Results

Bias means vspy minus Praat. COG and SD errors are in Hz; skew and excess
kurtosis are dimensionless. Correlations are across segments.

| Averaging region | Moment | N | Bias | MAE | RMSE | Pearson r |
|---|---|---:|---:|---:|---:|---:|
| full | cog | 271 | -561.1546 | 657.3475 | 788.7565 | 0.9167 |
| full | sdev | 271 | -147.7863 | 254.7771 | 418.6597 | 0.7061 |
| full | skew | 271 | 0.4250 | 0.5131 | 0.7089 | 0.8193 |
| full | kurt | 271 | 2.3620 | 2.8818 | 9.7761 | 0.1534 |
| middle80 | cog | 271 | -135.0088 | 325.3338 | 508.8095 | 0.9307 |
| middle80 | sdev | 271 | -145.7065 | 212.3052 | 382.5305 | 0.7689 |
| middle80 | skew | 271 | 0.1686 | 0.3158 | 0.4974 | 0.8814 |
| middle80 | kurt | 271 | 1.2927 | 1.8224 | 7.5208 | 0.2400 |

## Interpretation and reproduction

This is a comparison of two aggregation methods, not a numerical equivalence
test. Praat averages unnormalized power spectra from six windows, then computes
moments. vspy computes normalized moments every millisecond, then averages each
moment equally across frames. These operations generally differ, particularly
for variance, skewness, and kurtosis, and for time-varying intensity. The script
also filters the extracted interval before windowing, uses a 0.1 ms window
extension and one-based cutoff-bin selection; vspy excludes bins below 300 Hz
on a 15 ms NumPy Hann spectrum. These implementation differences are retained.

Run `python -m test.compare_spectral_moments` from the repository root.
Raw Praat logs are in praat_raw/, vspy frame outputs in frames/, paired interval
results in segment_comparison.csv, and aggregate results in metrics.csv.
