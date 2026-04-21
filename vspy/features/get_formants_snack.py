# script used to get formant estimations in the same way as Snack 

# pre-emphasis --> downsample --> slice frame --> apply Hamming window --> fit LPC --> find peaks (roots)

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from math import gcd
from vspy.io import read_wav

# define LPC helper 
# fits a polynomial whos roots are resonances of vocal tract 
# compute frequency and bandwidth 
def _lpc_formants(frame, order, fs, max_bandwidth):
    window = np.hamming(len(frame))
    windowed = frame * window

    # compute autocorrelation and solve Yule-Walker equations for LPC coefficients
    # Levinson-Durbin LPC takes in autocorrelation as input
    # fitting an all-pole filter to the spectral envelope 
    r = np.array([np.dot(windowed[:len(windowed)-k], windowed[k:]) for k in range(order + 1)])
    if r[0] == 0:
        return [], []
    R = np.array([[r[abs(i-j)] for j in range(order)] for i in range(order)])
    try:
        a_rest = np.linalg.solve(R, -r[1:])
    except np.linalg.LinAlgError:
        return [], []
    a = np.concatenate([[1.0], a_rest])

    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]

    freqs = np.arctan2(np.imag(roots), np.real(roots)) * (fs / (2 * np.pi))
    bws   = -np.log(np.abs(roots)) * (fs / np.pi)

    formants = []
    bandwidths = []
    for f, b in sorted(zip(freqs, bws)):
        if 50 < f < fs / 2 and b < max_bandwidth:
            formants.append(f)
            bandwidths.append(b)

    return formants, bandwidths


# define function with hyperparamters 
def get_formants_snack(
    wavfile,
    frameshift_ms=1,
    datalen=None,
    num_formants=4,
    window_ms=25,
    pre_emphasis=0.98,
    # determines how many resonances a model can represent 
    # fs / 1000 + 2 (fs = sampling rate)
    # the audio is downsampled to 10kHz: 10000 / 1000 = 10 + 2 = 12
    lpc_order=12,
    ds_freq=10000,
    max_bandwidth=500,
):
    y, fs = read_wav(wavfile)

    # pre-emphasis: flatten spectral tilt so formant peaks are equal height
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # downsample to ds_freq — F1-F4 all sit below 5kHz, no need for full rate
    g = gcd(ds_freq, fs)
    y_ds = resample_poly(y, ds_freq // g, fs // g)
    fs_ds = ds_freq

    frame_step = int(round(frameshift_ms / 1000 * fs_ds))
    frame_size = int(round(window_ms / 1000 * fs_ds))
    n_frames   = (len(y_ds) - frame_size) // frame_step

    if datalen is None:
        datalen = n_frames

    F1 = np.full(datalen, np.nan)
    F2 = np.full(datalen, np.nan)
    F3 = np.full(datalen, np.nan)
    F4 = np.full(datalen, np.nan)
    B1 = np.full(datalen, np.nan)
    B2 = np.full(datalen, np.nan)
    B3 = np.full(datalen, np.nan)
    B4 = np.full(datalen, np.nan)

    for i in range(n_frames):
        start = i * frame_step
        frame = y_ds[start : start + frame_size]
        formants, bws = _lpc_formants(frame, lpc_order, fs_ds, max_bandwidth)

        t_ms = i * frame_step / fs_ds * 1000
        idx  = round(t_ms / frameshift_ms)
        if not (0 <= idx < datalen):
            continue

        if len(formants) > 0:
            F1[idx] = formants[0]
        if len(formants) > 1:
            F2[idx] = formants[1]
        if len(formants) > 2:
            F3[idx] = formants[2]
        if len(formants) > 3:
            F4[idx] = formants[3]
        if len(bws) > 0:
            B1[idx] = bws[0]
        if len(bws) > 1:
            B2[idx] = bws[1]
        if len(bws) > 2:
            B3[idx] = bws[2]
        if len(bws) > 3:
            B4[idx] = bws[3]

    t_ms = np.arange(datalen) * frameshift_ms
    return pd.DataFrame({
        "t_ms": t_ms,
        "F1_snack": F1, "F2_snack": F2, "F3_snack": F3, "F4_snack": F4,
        "B1_snack": B1, "B2_snack": B2, "B3_snack": B3, "B4_snack": B4,
    })
