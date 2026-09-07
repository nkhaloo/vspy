# Full vspy pipeline with VoiceSauce STRAIGHT F0

STRAIGHT is the default F0 source in vspy. This run computed all registered features and all downstream features on the 154 restored recordings. Every F0-dependent calculation used the Python STRAIGHT track; Snack supplied the driving formants.

## STRAIGHT precision

The separate audit covers all **69,452** VoiceSauce reference rows: **69,448/69,448** finite F0 values match after rounding to three decimals. Missing-value disagreements: **0** (4 missing reference values).

Unfiltered F0 MAE: **0.000248810474 Hz**; RMSE: **0.000288418159 Hz**; maximum absolute error: **0.000499999857 Hz**. These errors fall within the CSV rounding interval. This establishes agreement at the export precision, not bit identity with MATLAB's unavailable unrounded doubles.

## Existing evaluation methodology

The existing `test/evaluate.py` joined **69,298** non-silence rows by filename and `t_ms`. Its half-open TextGrid labels exclude one endpoint per recording relative to the inclusive VoiceSauce export. The separate F0 audit above includes those endpoints.

Pearson r, concordance correlation (CCC), mean/median error, MAE, coverage and counts use the existing pairwise finite-value filter and independent MAD outlier rejection (modified z-score > 3.5 on either system). Coverage below is retained pairs divided by merged rows. The CSVs also contain all outlier counts and signed errors. F0 VDE/GPE are calculated before MAD filtering. Displayed 0.000 errors and 1.0000 correlations are rounded, not claims of exact binary equality.

## F0 results

| Feature | Pearson r | CCC | MAE (Hz) | Coverage | VDE | GPE | FPE mean (Hz) | FPE SD (Hz) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f0_straight | 1.0000 | 1.0000 | 0.000 | 91.76% | 0.0000 | 0.0000 | -0.000 | 0.000 |
| f0_shr | 1.0000 | 1.0000 | 0.000 | 88.29% | 0.0000 | 0.0000 | -0.000 | 0.000 |
| f0_snack | 1.0000 | 1.0000 | 0.000 | 93.87% | 0.0000 | 0.0000 | 0.000 | 0.000 |
| f0_praat | 0.9964 | 0.9964 | 0.398 | 39.41% | 0.0100 | 0.0027 | 0.145 | 6.564 |

## All other feature results

Formant/bandwidth MAEs are in Hz; harmonic/tilt, CPP and HNR MAEs are in dB. Energy, SoE, SHR and epoch values retain their original units. Tilt names refer to corrected tilts, following the existing evaluator mappings.

| Feature | Pearson r | CCC | MAE | Coverage | Retained frames |
|---|---:|---:|---:|---:|---:|
| F1_praat | 0.9519 | 0.9518 | 51.655 | 93.80% | 64,999 |
| F2_praat | 0.9524 | 0.9523 | 51.299 | 93.79% | 64,996 |
| F3_praat | 0.9650 | 0.9650 | 51.090 | 92.55% | 64,134 |
| F4_praat | 0.9921 | 0.9921 | 21.864 | 40.20% | 27,855 |
| F1_snack | 0.9919 | 0.9918 | 1.796 | 96.81% | 67,086 |
| F2_snack | 0.9862 | 0.9862 | 6.054 | 97.27% | 67,406 |
| F3_snack | 0.9609 | 0.9609 | 6.853 | 92.43% | 64,052 |
| F4_snack | 0.9253 | 0.9251 | 12.542 | 96.49% | 66,869 |
| B1_praat | 0.9546 | 0.9546 | 46.651 | 89.17% | 61,795 |
| B2_praat | 0.9599 | 0.9599 | 46.032 | 89.36% | 61,928 |
| B3_praat | 0.9604 | 0.9604 | 51.652 | 87.96% | 60,953 |
| B4_praat | 0.9868 | 0.9868 | 32.540 | 37.92% | 26,276 |
| B1_snack | 0.9966 | 0.9966 | 0.589 | 91.60% | 63,479 |
| B2_snack | 0.9988 | 0.9988 | 0.304 | 89.97% | 62,347 |
| B3_snack | 0.9962 | 0.9962 | 0.735 | 91.77% | 63,595 |
| B4_snack | 0.9876 | 0.9874 | 2.955 | 95.43% | 66,130 |
| H1 | 0.9996 | 0.9995 | 0.252 | 93.62% | 64,880 |
| H2 | 0.9995 | 0.9994 | 0.274 | 93.59% | 64,859 |
| H4 | 0.9991 | 0.9990 | 0.377 | 93.62% | 64,880 |
| A1 | 0.9985 | 0.9985 | 0.270 | 92.71% | 64,248 |
| A2 | 0.9983 | 0.9982 | 0.308 | 92.91% | 64,387 |
| A3 | 0.9989 | 0.9989 | 0.322 | 92.85% | 64,341 |
| H2K | 0.9939 | 0.9933 | 0.774 | 93.49% | 64,788 |
| H5K | 0.9919 | 0.9911 | 0.810 | 92.72% | 64,254 |
| H1c | 0.9994 | 0.9993 | 0.258 | 92.91% | 64,387 |
| H2c | 0.9989 | 0.9988 | 0.294 | 92.89% | 64,374 |
| H4c | 0.9954 | 0.9954 | 0.443 | 92.60% | 64,169 |
| A1c | 0.9965 | 0.9965 | 0.313 | 92.17% | 63,871 |
| A2c | 0.9954 | 0.9953 | 0.392 | 92.24% | 63,920 |
| A3c | 0.9913 | 0.9912 | 0.499 | 91.94% | 63,716 |
| H2Kc | 0.9877 | 0.9877 | 0.945 | 91.77% | 63,594 |
| H1H2 | 0.9922 | 0.9922 | 0.284 | 92.46% | 64,076 |
| H2H4 | 0.9895 | 0.9893 | 0.490 | 92.91% | 64,387 |
| H1A1 | 0.9903 | 0.9903 | 0.344 | 92.91% | 64,385 |
| H1A2 | 0.9904 | 0.9904 | 0.452 | 91.48% | 63,394 |
| H1A3 | 0.9865 | 0.9864 | 0.576 | 92.84% | 64,333 |
| H4H2K | 0.9815 | 0.9815 | 0.974 | 91.96% | 63,729 |
| H2KH5K | 0.9867 | 0.9865 | 1.212 | 90.82% | 62,936 |
| CPP | 1.0000 | 1.0000 | 0.000 | 91.31% | 63,276 |
| Energy | 1.0000 | 1.0000 | 0.000 | 89.90% | 62,297 |
| HNR05 | 1.0000 | 1.0000 | 0.000 | 91.86% | 63,656 |
| HNR15 | 1.0000 | 1.0000 | 0.000 | 91.86% | 63,655 |
| HNR25 | 1.0000 | 1.0000 | 0.000 | 91.86% | 63,654 |
| HNR35 | 1.0000 | 1.0000 | 0.000 | 91.84% | 63,640 |
| SoE | 0.9999 | 0.9999 | 0.000 | 11.88% | 8,230 |
| epoch | 1.0000 | 1.0000 | 0.000 | 100.00% | 69,298 |
| SHR | 1.0000 | 1.0000 | 0.000 | 97.10% | 67,286 |

## Interpretation and coverage

F0 agreement does not make every downstream implementation identical. The tables retain the residual formant, bandwidth, harmonic and tilt errors. SoE has low coverage because it is defined only at detected epochs. Praat/SHR/Snack F0 are comparison tracks, not inputs to the dependent features in this run.

The saved reference supplies comparisons for every implemented main-pipeline output except F2K (the selected peak frequency in the 2-kHz region), which has no corresponding reference column. F2K is present in the full output CSV. The separate nasality API has no corresponding measurements in this VoiceSauce export.

## Validation and artifacts

- `tests.log`: regression tests against eight corpus recordings (including the cases sensitive to MATLAB frame-time arithmetic), a synthetic periodic signal, segmentation/frameshift behavior, and routing of STRAIGHT to every dependent feature.
- [Full output](output_straight.csv)
- [Feature metrics](feature_metrics.csv) and [F0 metrics](f0_metrics.csv)
- [Per-frame feature errors](per_frame_errors.csv)
- [All-reference F0 precision by recording](straight_all_reference_precision.csv) and [all reference frame comparisons](straight_all_reference_frames.csv)
- [Run manifest with audio, TextGrid and Python source hashes](run_metadata.json)
- [Implementation details and source provenance](../../docs/straight_port.md)

Runtime environment: Python 3.12.2, NumPy 1.26.4, SciPy 1.17.0. Full-pipeline wall time: 121.2 seconds.

Reproduction from the repository root:

```sh
python -m test.runner --jobs 4
python -m test.validate_straight_precision --jobs 4
python -m pytest test/test_straight_port.py -q
python -m test.report_straight_results
```
