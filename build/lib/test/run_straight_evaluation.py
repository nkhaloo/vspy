"""Run the saved VoiceSauce corpus through vspy, with STRAIGHT driving F0.

From the repository root: python -m test.run_straight_evaluation
The reference selects filenames only; its feature values are never supplied
as inputs to vspy. Existing per-file outputs can be resumed with --resume.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from vspy import vspy
from test.evaluate import run


def _process_recording(job):
    path, textgrid_dir, dest, resume = job
    if resume and dest.exists():
        return pd.read_csv(dest)
    return vspy([path], textgrid_dir=textgrid_dir, tier='words',
                output_file=dest, f0_source='straight', formant_source='snack')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('test/output_vs.csv'))
    parser.add_argument('--wav-dir', type=Path, default=Path('test/wav'))
    parser.add_argument('--textgrid-dir', type=Path, default=Path('test/textgrid'))
    parser.add_argument('--outdir', type=Path, default=Path('test/straight_results'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    reference = pd.read_csv(args.reference)
    names = sorted(reference.Filename.unique())
    paths = [args.wav_dir / (Path(name).stem + '.wav') for name in names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    args.outdir.mkdir(parents=True, exist_ok=True)
    per_file = args.outdir / 'files'
    per_file.mkdir(exist_ok=True)
    frames = []
    start = time.monotonic()
    jobs = [(p, args.textgrid_dir, per_file / (p.stem + '.csv'), args.resume) for p in paths]
    pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    try:
        results = pool.map(_process_recording, jobs) if pool else map(_process_recording, jobs)
        for i, (path, frame) in enumerate(zip(paths, results), 1):
            frames.append(frame)
            print(f'{i}/{len(paths)} {path.name}: {len(frame)} frames ({time.monotonic()-start:.1f}s)', flush=True)
    finally:
        if pool:
            pool.shutdown()
    combined = pd.concat(frames, ignore_index=True)
    output = args.outdir / 'output_straight.csv'
    combined.to_csv(output, index=False)
    run(output, args.reference, args.outdir)
    # Unfiltered accuracy, including the endpoints excluded by old labels.
    combined['_stem'] = combined.filename.str.rsplit('.', n=1).str[0]
    reference['_stem'] = reference.Filename.str.rsplit('.', n=1).str[0]
    merged = combined.merge(reference, on=['_stem', 't_ms'], validate='one_to_one')
    records = []
    for name, frame in list(merged.groupby('_stem')) + [('ALL', merged)]:
        a, b = frame.f0_straight.to_numpy(), frame.strF0.to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        error = a[finite] - b[finite]
        records.append(dict(filename=name, frames=len(frame), finite_pairs=int(finite.sum()),
                            missing_disagreements=int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
                            exact_double_matches=int(np.count_nonzero(a[finite] == b[finite])),
                            matches_at_3_decimals=int(np.count_nonzero(np.round(a[finite], 3) == b[finite])),
                            mae_hz=float(np.mean(np.abs(error))), max_abs_error_hz=float(np.max(np.abs(error))),
                            rmse_hz=float(np.sqrt(np.mean(error**2)))))
    pd.DataFrame(records).to_csv(args.outdir / 'straight_precision.csv', index=False)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    code_paths = sorted(Path('vspy').rglob('*.py')) + [
        Path('test/evaluate.py'), Path(__file__), Path('test/runner.py')]
    textgrids = [args.textgrid_dir / (p.stem + '.TextGrid') for p in paths]
    metadata = dict(reference=str(args.reference), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                    audio_sha256=hashes, f0_source='straight', formant_source='snack',
                    code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
                    textgrid_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in textgrids},
                    f0_segmentation='words tier, 25 ms buffer, ignore empty/space/SIL/[SIL]',
                    frameshift_ms=1, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    elapsed_seconds=time.monotonic()-start)
    (args.outdir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(pd.DataFrame(records).tail(1).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
