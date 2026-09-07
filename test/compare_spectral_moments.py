"""Compare the original Praat script with framewise vspy moments on vowels.

Run: python -m test.compare_spectral_moments [--sanity-only] [--jobs 4] [--resume]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call, run_file
import soundfile as sf

from vspy import registry

ROOT = Path(__file__).resolve().parents[1]
MOMENTS = dict(cog='spectral_cog', sdev='spectral_sdev',
               skew='spectral_skew', kurt='spectral_kurt')
EXCLUDED = {'', 'bad', 'sil', '[sil]', 'silence', 'sp', 'pau'}


def process_file(job):
    wav, grid, group, out, resume = job
    rows, errors = [], []
    key = group + '__' + grid.stem
    try:
        tg = call('Read from file', str(grid))
        if isinstance(tg, list):
            is_collection = len(tg) > 1
            grids = [obj for obj in tg if obj.class_name == 'TextGrid']
            if len(grids) != 1:
                raise ValueError(f'Expected one TextGrid in collection, found {len(grids)}')
            tg = grids[0]
            if is_collection and tg.name != grid.stem:
                raise ValueError(f'Collection TextGrid is named {tg.name!r}, not {grid.stem!r}; audio pairing is ambiguous')
        if tg.class_name != 'TextGrid':
            raise ValueError(f'Expected TextGrid, found {tg.class_name}')
        targets = []
        for tier in range(1, call(tg, 'Get number of tiers') + 1):
            name = call(tg, 'Get tier name', tier)
            if name.strip().lower() not in ('vowel', 'vowels') or not call(tg, 'Is interval tier', tier):
                continue
            intervals = []
            for j in range(1, call(tg, 'Get number of intervals', tier) + 1):
                label = call(tg, 'Get label of interval', tier, j)
                if label.strip().lower() not in EXCLUDED:
                    start = call(tg, 'Get start time of interval', tier, j)
                    end = call(tg, 'Get end time of interval', tier, j)
                    if end <= start:
                        errors.append(dict(file=str(grid.relative_to(ROOT)), tier=tier,
                                           interval=j, stage='annotation', error='Nonpositive interval duration'))
                        call(tg, 'Set interval text', tier, j, '')
                        continue
                    intervals.append((j, start, end, label))
                    call(tg, 'Set interval text', tier, j, 'vowel')
                else:
                    call(tg, 'Set interval text', tier, j, '')
            if intervals:
                targets.append((tier, name, intervals))
        if not targets:
            return rows, errors, False
        info = sf.info(wav)
        frame_path = out / 'frames' / (key + '.csv.gz')
        legacy_frame_path = out / 'frames' / (key + '.csv')
        if resume and frame_path.exists():
            frames = pd.read_csv(frame_path)
        elif resume and legacy_frame_path.exists():
            frames = pd.read_csv(legacy_frame_path)
        else:
            frames = registry.run(['spectral_moments'], wav, 1,
                                  int(info.frames / info.samplerate * 1000))['spectral_moments']
            frames.insert(0, 't_ms', np.arange(len(frames)))
            frames.to_csv(frame_path, index=False)
        for tier, name, intervals in targets:
            try:
                raw_path = out / 'praat_raw' / f'{key}_tier{tier}.txt'
                if resume and raw_path.exists():
                    raw = raw_path.read_text()
                else:
                    with tempfile.TemporaryDirectory() as directory:
                        dest = Path(directory)
                        shutil.copyfile(wav, dest / 'audio.wav')
                        call(tg, 'Save as text file', str(dest / 'audio.TextGrid'))
                        run_file(str(ROOT / 'spectral moments'), directory + '/',
                                 'vowel', 'log', tier, info.samplerate, 6, .015, 300)
                        raw = (dest / 'log.txt').read_text()
                    raw_path.write_text(raw)
                reference = pd.read_csv(io.StringIO(raw), sep='\t').dropna(subset=['start'])
                if len(reference) != len(intervals):
                    if resume:
                        return process_file((wav, grid, group, out, False))
                    raise ValueError('Praat output interval count does not match TextGrid')
                for (_, ref), (index, start, end, label) in zip(reference.iterrows(), intervals):
                    if not np.isclose(ref.start, start):
                        raise ValueError('Praat output start does not match TextGrid')
                    duration = end - start
                    row = dict(dataset=group, filename=wav.name, tier=name, tier_index=tier,
                               interval=index, label=label, start_s=start, end_s=end,
                               duration_s=duration, sample_rate=info.samplerate,
                               excessive_window_overlap=6 * .015 > 1.6 * duration,
                               praat_intensity=ref.intensity)
                    for moment in MOMENTS:
                        row['praat_' + moment] = pd.to_numeric(ref[moment], errors='coerce')
                    for scope, lo, hi in [('full', start, end),
                                           ('middle80', start + .1 * duration, end - .1 * duration)]:
                        selected = frames.loc[(frames.t_ms >= lo * 1000) & (frames.t_ms < hi * 1000)]
                        row[scope + '_frame_count'] = len(selected)
                        row[scope + '_valid_frames'] = int(selected.spectral_cog.notna().sum())
                        for moment, column in MOMENTS.items():
                            value = selected[column].mean()
                            row[f'vspy_{scope}_{moment}'] = value
                            row[f'delta_{scope}_{moment}'] = value - row['praat_' + moment]
                    rows.append(row)
            except Exception as exc:
                errors.append(dict(file=str(grid.relative_to(ROOT)), tier=tier, stage='praat', error=str(exc)))
        return rows, errors, True
    except Exception as exc:
        errors.append(dict(file=str(grid.relative_to(ROOT)), stage='read/extract', error=str(exc)))
        return rows, errors, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanity-only', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse raw logs and frames from an identical prior run')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'test/spectral_moments_vowel_results')
    args = parser.parse_args()
    out = args.outdir.resolve()
    out.mkdir(exist_ok=True, parents=True)
    (out / 'praat_raw').mkdir(exist_ok=True)
    (out / 'frames').mkdir(exist_ok=True)
    jobs = []
    if not args.sanity_only:
        audio = {p.stem: p for p in (ROOT / 'test/wav').iterdir()
                 if p.suffix.lower() == '.wav'}
        for grid in sorted((ROOT / 'test/textgrid').iterdir()):
            if grid.suffix.lower() != '.textgrid':
                continue
            wav = audio.get(grid.stem)
            if wav is not None:
                jobs.append((wav, grid, 'corpus', out, args.resume))
    for grid in sorted((ROOT / 'test/sanity test').rglob('*.TextGrid')):
        wav = grid.with_suffix('.wav')
        if wav.exists():
            jobs.append((wav, grid, 'sanity', out, args.resume))
    rows, errors = [], []
    file_count = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (results, failures, processed) in enumerate(pool.map(process_file, jobs), 1):
            rows.extend(results)
            errors.extend(failures)
            file_count += int(processed)
            if i % 100 == 0:
                print(f'Scanned {i}/{len(jobs)} files; extracted {file_count}; compared {len(rows)} vowels', flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / 'segment_comparison.csv', index=False)
    identifiers = ['dataset', 'filename', 'tier', 'tier_index', 'interval',
                   'label', 'start_s', 'end_s', 'duration_s', 'sample_rate']
    comparison[identifiers + ['praat_intensity'] +
               [f'praat_{m}' for m in MOMENTS]].to_csv(out / 'praat_segments.csv', index=False)
    comparison[identifiers + [c for c in comparison if c.startswith('vspy_') or
                              c.endswith('_frame_count') or c.endswith('_valid_frames')]].to_csv(
                                  out / 'vspy_segment_means.csv', index=False)
    summary = []
    for dataset in ['all'] + sorted(comparison.dataset.unique().tolist()):
        subset = comparison if dataset == 'all' else comparison[comparison.dataset == dataset]
        for scope in ('full', 'middle80'):
            for moment in MOMENTS:
                pair = subset[[f'praat_{moment}', f'vspy_{scope}_{moment}']].replace([np.inf, -np.inf], np.nan).dropna()
                ref, actual = pair.iloc[:, 0], pair.iloc[:, 1]
                delta = actual - ref
                summary.append(dict(dataset=dataset, scope=scope, moment=moment, n=len(pair),
                                    praat_mean=ref.mean(), vspy_mean=actual.mean(),
                                    bias=delta.mean(), mae=delta.abs().mean(),
                                    rmse=np.sqrt((delta ** 2).mean()), correlation=ref.corr(actual)))
    metrics = pd.DataFrame(summary)
    metrics.to_csv(out / 'metrics.csv', index=False)
    (out / 'errors.json').write_text(json.dumps(errors, indent=2))
    metadata = dict(audio_files=file_count, compared_intervals=len(rows),
                    selection='nonempty vowel/vowels tiers, excluding bad and silence labels',
                    excluded_labels=sorted(EXCLUDED), hop_ms=1, window_ms=15,
                    cutoff_hz=300, praat_windows=6, praat_script='spectral moments (unchanged)',
                    praat_sha256=hashlib.sha256((ROOT / 'spectral moments').read_bytes()).hexdigest(),
                    resampling='original sample rate per file',
                    parselmouth_version=parselmouth.__version__, praat_version=parselmouth.PRAAT_VERSION)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    report = f'''# Vowel spectral moments comparison

Compared {len(rows)} vowel intervals from {file_count} audio files. Selection:
nonempty intervals on explicitly named vowel/vowels tiers in test/textgrid and
test/sanity test; labels marked bad or silence are excluded. Generic segment
tiers are not guessed to contain vowels. Original labels are preserved in the
comparison CSV. Labels on vowel tiers are treated as vowel annotations even
when they describe phonation rather than vowel identity.

The original Praat script was executed unchanged through Parselmouth. Temporary
TextGrid copies relabel selected intervals to `vowel`, allowing the original
script's exact-label selection to process them together. Original TextGrids
are unchanged. Form arguments use the annotated tier and original sample rate
(no upsampling). Defaults retained: six 15 ms windows, 300 Hz cutoff, middle
80% of each interval. This runs the script's fricative-oriented settings on
vowels as requested; it does not establish that those settings suit every
vowel-analysis research question.

vspy's registered spectral_moments extractor ran at a 1 ms hop on the original
audio. Arithmetic means of valid frame moments were calculated separately for
frame centers in [start, end) and in the middle 80% of each interval. NaNs are
excluded; frame counts are in segment_comparison.csv. Frames are computed on
the continuous recording, so windows near segment boundaries may cross them.

{int(comparison.excessive_window_overlap.sum())} intervals violate the original
script's recommendation that six window lengths total no more than 1.6 times
the segment duration; these are flagged and retained. {len(errors)} read/run
or annotation issues are recorded in errors.json. Undefined comparison values
remain in the segment CSV and are excluded pairwise from the metrics below.

## Results

Bias means vspy minus Praat. COG and SD errors are in Hz; skew and excess
kurtosis are dimensionless. Correlations are across segments.

| Dataset | Averaging region | Moment | N | Bias | MAE | RMSE | Pearson r |
|---|---|---|---:|---:|---:|---:|---:|
'''
    for m in summary:
        report += f"| {m['dataset']} | {m['scope']} | {m['moment']} | {m['n']} | {m['bias']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['correlation']:.4f} |\n"
    report += '''
## Interpretation and reproduction

This compares two aggregation methods, not numerical equivalence. Praat
averages unnormalized power spectra from six windows, then computes moments.
vspy computes normalized moments every millisecond, then averages each moment
equally across frames. These operations generally differ, particularly for
variance, skewness, and kurtosis, and for time-varying intensity. The script
also filters the extracted interval before windowing, uses a 0.1 ms window
extension and one-based cutoff-bin selection; vspy excludes bins below 300 Hz
on a 15 ms NumPy Hann spectrum. These implementation differences are retained.

Run `python -m test.compare_spectral_moments` from the repository root (or add
`--sanity-only` for just the three sanity recordings). Raw Praat logs are in
praat_raw/, compressed vspy frame CSVs in frames/, paired interval results in
segment_comparison.csv, and aggregate results in metrics.csv. Standalone
segment tables are praat_segments.csv and vspy_segment_means.csv. Use --resume
only when the input audio, annotations, script, and extraction settings are
unchanged; it reuses previously saved raw logs and frames.
'''
    (out / 'report.md').write_text(report)
    print(metadata)
    print(metrics.to_string(index=False))
    print(f'Errors: {len(errors)}; output: {out}')


if __name__ == '__main__':
    main()
