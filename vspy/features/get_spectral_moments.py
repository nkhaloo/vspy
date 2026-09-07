"""Framewise power-weighted moments, adapted from DiCanio's Praat script.

Uses its moment definitions (including excess kurtosis), but a single moving
window instead of averaging six spectra over a labeled fricative interval.
"""

import numpy as np
import pandas as pd

from vspy.io import read_wav


def get_spectral_moments(wavfile, frameshift_ms=1, datalen=None,
                         window_ms=15, high_pass_cutoff=300):
    """Return COG/SD in Hz and dimensionless skewness/excess kurtosis.

    Frames are centered at k * frameshift_ms, rounded to the nearest sample.
    Incomplete windows and zero-power spectra produce NaNs. Frequencies below
    high_pass_cutoff are excluded. No resampling or voicing mask is applied.
    """
    if not np.isfinite(frameshift_ms) or frameshift_ms <= 0:
        raise ValueError("frameshift_ms must be finite and positive")
    if not np.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("window_ms must be finite and positive")
    y, fs = read_wav(wavfile)
    if not np.isfinite(high_pass_cutoff) or not 0 <= high_pass_cutoff < fs / 2:
        raise ValueError("high_pass_cutoff must be between zero and Nyquist")
    if datalen is None:
        datalen = int(len(y) / fs * 1000 / frameshift_ms)
    size = int(np.floor(window_ms * fs / 1000 + 0.5))
    if size < 3:
        raise ValueError("window_ms must span at least three samples")
    frequencies = np.fft.rfftfreq(size, 1 / fs)
    keep = frequencies >= high_pass_cutoff
    frequencies = frequencies[keep]
    window = np.hanning(size)
    values = np.full((datalen, 4), np.nan)
    for k in range(datalen):
        center = int(np.floor(k * frameshift_ms * fs / 1000 + 0.5))
        start = center - size // 2
        if start < 0 or start + size > len(y):
            continue
        power = np.abs(np.fft.rfft(y[start:start + size] * window))[keep] ** 2
        total = power.sum()
        if not np.isfinite(total) or total <= 0:
            continue
        weights = power / total
        cog = np.sum(frequencies * weights)
        delta = frequencies - cog
        variance = np.sum(delta ** 2 * weights)
        values[k, :2] = cog, np.sqrt(variance)
        if variance > 0:
            values[k, 2] = np.sum(delta ** 3 * weights) / variance ** 1.5
            values[k, 3] = np.sum(delta ** 4 * weights) / variance ** 2 - 3
    return pd.DataFrame(values, columns=[
        "spectral_cog", "spectral_sdev", "spectral_skew", "spectral_kurt",
    ])
