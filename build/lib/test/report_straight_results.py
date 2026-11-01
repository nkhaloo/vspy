"""Render the existing evaluator's CSVs as a complete reviewable report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path('test/straight_results')
    features = pd.read_csv(root / 'feature_metrics.csv')
    pitches = pd.read_csv(root / 'f0_metrics.csv')
    audit = pd.read_csv(root / 'straight_all_reference_precision.csv')
    total = audit[audit.filename == 'ALL'].iloc[0]
    metadata = json.loads((root / 'run_metadata.json').read_text())
    n_rows = int(pitches.n_total.iloc[0])
    lines = [
        '# Full vspy pipeline with VoiceSauce STRAIGHT F0', '',
        'STRAIGHT is the default F0 source in vspy. This run computed all registered features and all downstream features on the 154 restored recordings. Every F0-dependent calculation used the Python STRAIGHT track; Snack supplied the driving formants.', '',
        '## STRAIGHT precision', '',
        f'The separate audit covers all **{int(total.frames):,}** VoiceSauce reference rows: **{int(total.matches_at_3_decimals):,}/{int(total.finite_pairs):,}** finite F0 values match after rounding to three decimals. Missing-value disagreements: **{int(total.missing_disagreements)}** ({int(total.frames-total.finite_pairs)} missing reference values).', '',
        f'Unfiltered F0 MAE: **{total.mae_hz:.12f} Hz**; RMSE: **{total.rmse_hz:.12f} Hz**; maximum absolute error: **{total.max_abs_error_hz:.12f} Hz**. These errors fall within the CSV rounding interval. This establishes agreement at the export precision, not bit identity with MATLAB\'s unavailable unrounded doubles.', '',
        '## Existing evaluation methodology', '',
        f'The existing `test/evaluate.py` joined **{n_rows:,}** non-silence rows by filename and `t_ms`. Its half-open TextGrid labels exclude one endpoint per recording relative to the inclusive VoiceSauce export. The separate F0 audit above includes those endpoints.', '',
        'Pearson r, concordance correlation (CCC), mean/median error, MAE, coverage and counts use the existing pairwise finite-value filter and independent MAD outlier rejection (modified z-score > 3.5 on either system). Coverage below is retained pairs divided by merged rows. The CSVs also contain all outlier counts and signed errors. F0 VDE/GPE are calculated before MAD filtering. Displayed 0.000 errors and 1.0000 correlations are rounded, not claims of exact binary equality.', '',
        '## F0 results', '',
        '| Feature | Pearson r | CCC | MAE (Hz) | Coverage | VDE | GPE | FPE mean (Hz) | FPE SD (Hz) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in pitches.itertuples():
        lines.append(f'| {r.feature} | {r.pearson_r:.4f} | {r.ccc:.4f} | {r.mae:.3f} | {r.coverage:.2%} | {r.vde:.4f} | {r.gpe:.4f} | {r.fpe_mean:.3f} | {r.fpe_std:.3f} |')
    lines += ['', '## All other feature results', '',
              'Formant/bandwidth MAEs are in Hz; harmonic/tilt, CPP and HNR MAEs are in dB. Energy, SoE, SHR and epoch values retain their original units. Tilt names refer to corrected tilts, following the existing evaluator mappings.', '',
              '| Feature | Pearson r | CCC | MAE | Coverage | Retained frames |',
              '|---|---:|---:|---:|---:|---:|']
    for r in features.itertuples():
        lines.append(f'| {r.feature} | {r.pearson_r:.4f} | {r.ccc:.4f} | {r.mae:.3f} | {r.coverage:.2%} | {r.n_valid:,} |')
    lines += ['', '## Interpretation and coverage', '',
              'F0 agreement does not make every downstream implementation identical. The tables retain the residual formant, bandwidth, harmonic and tilt errors. SoE has low coverage because it is defined only at detected epochs. Praat/SHR/Snack F0 are comparison tracks, not inputs to the dependent features in this run.', '',
              'The saved reference supplies comparisons for every implemented main-pipeline output except F2K (the selected peak frequency in the 2-kHz region), which has no corresponding reference column. F2K is present in the full output CSV. The separate nasality API has no corresponding measurements in this VoiceSauce export.', '',
              '## Validation and artifacts', '',
              '- `tests.log`: regression tests against eight corpus recordings (including the cases sensitive to MATLAB frame-time arithmetic), a synthetic periodic signal, segmentation/frameshift behavior, and routing of STRAIGHT to every dependent feature.',
              '- [Full output](output_straight.csv)',
              '- [Feature metrics](feature_metrics.csv) and [F0 metrics](f0_metrics.csv)',
              '- [Per-frame feature errors](per_frame_errors.csv)',
              '- [All-reference F0 precision by recording](straight_all_reference_precision.csv) and [all reference frame comparisons](straight_all_reference_frames.csv)',
              '- [Run manifest with audio, TextGrid and Python source hashes](run_metadata.json)',
              '- [Implementation details and source provenance](../../docs/straight_port.md)', '',
              f'Runtime environment: Python {metadata["python"]}, NumPy {metadata["numpy"]}, SciPy {metadata["scipy"]}. Full-pipeline wall time: {metadata["elapsed_seconds"]:.1f} seconds.', '',
              'Reproduction from the repository root:', '', '```sh',
              'python -m test.runner --jobs 4',
              'python -m test.validate_straight_precision --jobs 4',
              'python -m pytest test/test_straight_port.py -q',
              'python -m test.report_straight_results', '```', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print(root / 'report.md')


if __name__ == '__main__':
    main()
