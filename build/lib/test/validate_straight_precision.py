"""Audit F0 against every reference row, including label endpoints.

python -m test.validate_straight_precision --jobs 4
Feature evaluation remains in test/evaluate.py; this checks CSV precision.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf

from vspy.api import _resample
from vspy.align.textgrid import parse_textgrid
from vspy.features.get_pitch_straight import get_pitch_straight
from vspy.io import read_wav


def extract(stem):
    y, fs = read_wav(Path('test/wav') / (stem + '.wav'))
    y, fs = _resample(y, fs)
    intervals = parse_textgrid(Path('test/textgrid') / (stem + '.TextGrid'), 'words', round_times=False)
    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / 'input.wav'
        sf.write(wav, y, fs, subtype='PCM_16')
        return get_pitch_straight(wav, intervals=intervals, ignore_labels=('', ' ', 'SIL', '[SIL]'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    reference = pd.read_csv('test/output_vs.csv')
    groups = list(reference.groupby('Filename', sort=True))
    stems = [Path(name).stem for name, _ in groups]
    pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    records, comparisons = [], []
    try:
        outputs = pool.map(extract, stems) if pool else map(extract, stems)
        for (name, rows), f0 in zip(groups, outputs):
            times = rows.t_ms.to_numpy(int)
            actual = f0[times - 1]  # VoiceSauce writes element s at time s.
            comparisons.append(pd.DataFrame({'filename': name, 't_ms': times,
                                             'python_f0': actual, 'voicesauce_f0': rows.strF0.to_numpy()}))
    finally:
        if pool:
            pool.shutdown()
    comparisons = pd.concat(comparisons, ignore_index=True)
    for name, rows in list(comparisons.groupby('filename')) + [('ALL', comparisons)]:
        a, b = rows.python_f0.to_numpy(), rows.voicesauce_f0.to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        diff = a[finite] - b[finite]
        records.append(dict(filename=name, frames=len(rows), finite_pairs=int(finite.sum()),
                            missing_disagreements=int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
                            exact_double_matches=int(np.count_nonzero(a[finite] == b[finite])),
                            matches_at_3_decimals=int(np.count_nonzero(np.round(a[finite], 3) == b[finite])),
                            mae_hz=float(np.mean(np.abs(diff))), max_abs_error_hz=float(np.max(np.abs(diff))),
                            rmse_hz=float(np.sqrt(np.mean(diff**2)))))
    outdir = Path('test/straight_results')
    outdir.mkdir(exist_ok=True)
    comparisons.to_csv(outdir / 'straight_all_reference_frames.csv', index=False)
    pd.DataFrame(records).to_csv(outdir / 'straight_all_reference_precision.csv', index=False)
    print(pd.DataFrame(records).tail(1).to_string(index=False))


if __name__ == '__main__':
    main()
