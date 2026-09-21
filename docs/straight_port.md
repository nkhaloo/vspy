# VoiceSauce STRAIGHT F0 port

`vspy()` defaults to `f0_source='straight'`. The implementation is Python/NumPy/SciPy; it has no MATLAB runtime dependency and never reads reference feature values during extraction. The selected F0 is also added when the caller supplies a feature subset, so downstream computations cannot silently substitute another pitch estimator.

The same STRAIGHT array drives harmonic amplitudes, formant-associated amplitudes, 2/5-kHz amplitudes, corrected amplitudes and spectral tilts, CPP, energy, HNR, and SoE. Praat, Snack and SHR pitch are independent comparison outputs. Formant selection remains separately configurable; the corpus evaluation uses Snack formants, matching VoiceSauce's default.

## Source and behavior

The algorithm is transcribed from Hideki Kawahara's `TandemSTRAIGHTmonolithicPackage014`, supplied locally, with wrapper behavior taken from `VoiceSauce/func_StraightPitch.m` and frame export behavior from `VoiceSauce/vs_OutputToText.m`.

| MATLAB routine | Python implementation |
|---|---|
| `TandemSTRAIGHTGeneralBody`, `tandem2STRAIGHT`, `interp1H` | `_spectrum` |
| `TANDEMf0GB` | `_periodicity`, candidate peak interpolation in `straight_f0` |
| `exF0candidatesTSTRAIGHTGB` | `straight_f0`, `_calibration`, `_decimate`, `_frame_times` |
| `baseF0withIF`, `refineF0AssumptionGB` | `_base_if`, `_refine` |
| F0-producing portion of `autoF0Tracking` | `_track` |
| `func_StraightPitch` | `straight_f0` smoothing and `get_pitch_straight` segmentation |

Compatibility details include:

- 40–500 Hz default search bounds, 1-ms internal frames, three harmonics for two passes of refinement, and calibrated candidate scores.
- MATLAB's end-anchored IIR decimation phase, reflection length, rounding away from zero, conjugate transpose, and symmetric construction of colon frame times. Last-bit time differences can affect sample selection and tracker decisions.
- VoiceSauce's nine-tap smoothing with replicated endpoints. VoiceSauce computes a separate voicing decision but does **not** use it to mask F0; the Python F0 path preserves this. The independent `V` output is not exposed.
- TextGrid segments preserve fractional-millisecond boundaries before buffering by 25 ms. Segment extraction, overwrite order, truncation and placement use VoiceSauce's one-based indexing.
- The low-level `get_pitch_straight` array corresponds to MATLAB array elements. Registry alignment places element 1 at `t_ms=1`, matching VoiceSauce's CSV export and vspy's downstream window centers. Frame zero is empty.
- vspy's existing resample-to-16-kHz and PCM16 write/read preprocessing remains in place.

When `textgrid_dir` and `tier` are supplied, STRAIGHT automatically analyzes all intervals on the selected tier. Those intervals also determine the output labels. The main pipeline does not apply any STRAIGHT-specific label exclusions.

Without TextGrid segmentation, long recordings are analyzed continuously. VoiceSauce's long-recording branch uses the F0 vector in an audio sample-index expression (`floor(F0 / 1000)`); that defective branch is not reproduced. Empty, silent and very short signals return empty or missing F0 safely. Compatibility results below apply to the tested speech corpus, not these degenerate or long-recording cases.

## Reproduce the evaluation

From the repository root, with dependencies installed:

```sh
python -m test.runner
python -m test.validate_straight_precision --jobs 4
python -m pytest test/test_straight_port.py -q
```

`test.runner` selects only the 154 filenames present in `test/output_vs.csv`, computes **all registered features and all downstream features**, and calls the existing `test/evaluate.py` implementation. That evaluator retains its non-silence merge, Pearson correlation, concordance correlation, independent MAD outlier removal, coverage/error metrics and F0 VDE/GPE metrics. Its mapping now includes STRAIGHT, SHR pitch, fourth-formant bandwidths and epochs.

The default output directory is `test/straight_results`. It contains the complete output, per-recording outputs, the existing metric CSVs, unfiltered STRAIGHT precision checks and a reproducibility manifest. `--resume` reuses per-recording outputs and is only appropriate when code and run settings have not changed.

The separate precision audit covers every VoiceSauce reference row, including segment endpoints excluded by the existing evaluator's half-open TextGrid labels. It does not replace the correlation evaluation. `--jobs 4` can also be supplied to `test.runner` to process independent recordings in four worker processes.

The supplied VoiceSauce CSV has three decimal places. Agreement at this precision is measurable; bit identity against MATLAB's unrounded internal doubles cannot be established from this CSV. The report records precision agreement separately from correlation.

## Upstream attribution

TANDEM-STRAIGHT was designed and coded by Hideki Kawahara. The local package's `READMEfirst.txt` states that academic institutes may use it for research/education and directs non-academic institutes to Kansai TLO for licensing. The port retains that provenance; it is not the Apache-licensed legacy STRAIGHT implementation. VoiceSauce's wrapper is attributed to Yen-Liang Shue / UCLA SPAPL (2009–2016).
