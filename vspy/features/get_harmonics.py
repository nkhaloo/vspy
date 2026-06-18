# calculate harmonic peaks and their amplitudes 

import numpy as np
from scipy.optimize import minimize

# function that returns harmonic amplitudes
def get_harmonics(y, Fs, F0, n_periods=3):
    n_frames = len(F0)
    H1 = np.full(n_frames, np.nan)
    H2 = np.full(n_frames, np.nan)
    H4 = np.full(n_frames, np.nan)

    # df_amplitude computes a DFT at an arbitrary frequecy, giving you its amplitude
    # does this at different frequencies across a 10% interval around the theoretical harmonic 
    # optimizer tells it where to compute frequencies
    def dft_amplitude(segment, freq):
        n = np.arange(len(segment))
        v = np.exp(-1j * 2 * np.pi * freq * n / Fs)
        return 20 * np.log10(np.abs(segment @ v))
    # find_peak mimics VoiceSauce's func_GetHarmonics: a local Nelder-Mead
    # search starting at f_est, bounded to f_est +/- round(f_est*0.1) Hz
    # (VS uses fminsearchbnd, which wraps fminsearch/Nelder-Mead with bounds
    # via a transform and starts from x0=f_est -- unlike scipy's bounded
    # Brent method, which ignores f_est and searches the whole bracket).
    def find_peak(segment, f_est):
        df_range = round(f_est * 0.1)
        f_min = f_est - df_range
        f_max = f_est + df_range
        result = minimize(
            lambda f: -dft_amplitude(segment, f[0]),
            x0=[f_est],
            method='Nelder-Mead',
            bounds=[(f_min, f_max)],
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
