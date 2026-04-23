# calculate harmonic peaks and their amplitudes 

import numpy as np
from scipy.optimize import minimize_scalar

# function that returns harmonic amplitudes
def get_harmonics(y, Fs, F0, n_periods=3):
    n_frames = len(F0)
    H1 = np.full(n_frames, np.nan)
    H2 = np.full(n_frames, np.nan)
    H4 = np.full(n_frames, np.nan)

    # df_amplitude computes a DFT at an arbitrary frequecy, giving you its amplitude
    def dft_amplitude(segment, freq):
        n = np.arange(len(segment))
        v = np.exp(-1j * 2 * np.pi * freq * n / Fs)
        return 20 * np.log10(np.abs(segment @ v))
    # find_peak calls dft_amplitude to find the maximum peak at a +/- 10% interval around f_est (estimated harmonic)
    def find_peak(segment, f_est):
        f_min = f_est * 0.9
        f_max = f_est * 1.1
        result = minimize_scalar(
            lambda f: -dft_amplitude(segment, f),
            bounds=(f_min, f_max),
            method='bounded'
        )
        return -result.fun

    sampleshift = Fs / 1000  # samples per ms (1ms hop)

    #frame loop 
    for k in range(n_frames):
        f0 = F0[k]

        if np.isnan(f0) or f0 == 0:
            continue

        ks = round(k * sampleshift)
        N0 = Fs / f0  # samples per period

        ystart = round(ks - (n_periods / 2) * N0)
        yend   = round(ks + (n_periods / 2) * N0)

        if ystart < 0 or yend > len(y):
            continue

        segment = y[ystart:yend]

        # these get fed into the dft peak finder 
        H1[k] = find_peak(segment, f0)
        H2[k] = find_peak(segment, 2 * f0)
        H4[k] = find_peak(segment, 4 * f0)

    return H1, H2, H4
