"""Plot the saved vowel comparison (requires matplotlib)."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', nargs='?', type=Path,
                        default=Path('test/spectral_moments_vowel_results'))
    args = parser.parse_args()
    data = pd.read_csv(args.outdir / 'segment_comparison.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout='constrained')
    labels = [('cog', 'Center of gravity (Hz)', 100),
              ('sdev', 'Standard deviation (Hz)', 100),
              ('skew', 'Skewness', 1), ('kurt', 'Excess kurtosis', 10)]
    for ax, (moment, title, threshold) in zip(axes.flat, labels):
        pair = data[[f'praat_{moment}', f'vspy_middle80_{moment}']].replace(
            [np.inf, -np.inf], np.nan).dropna()
        x, y = pair.iloc[:, 0], pair.iloc[:, 1]
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.scatter(x, y, s=6, alpha=.18, color='#246ca6', rasterized=True)
        ax.plot([low, high], [low, high], '--', color='#c14d35', linewidth=1)
        ax.set_xscale('symlog', linthresh=threshold)
        ax.set_yscale('symlog', linthresh=threshold)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(f'{title}\nr = {x.corr(y):.3f}; MAE = {(y-x).abs().mean():.2f}')
        ax.set_xlabel('Praat: moments of averaged spectra')
        ax.set_ylabel('vspy: mean of frame moments')
        ax.grid(alpha=.15)
    fig.suptitle('Vowels: middle-80% comparison\nSymmetric-log axes; dashed line = equality', fontsize=14)
    fig.savefig(args.outdir / 'middle80_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
