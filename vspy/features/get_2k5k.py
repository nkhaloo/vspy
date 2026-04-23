# estimate harmonic closest to 2kHz and 5kHz
import numpy as np
from scipy.optimize import minimize_scalar


def get_2k5k(y, Fs, F0, n_periods=3):
    n_frames = len(F0)
    # empty arrays for harmonics
    H2K = np.full(n_frames, np.nan)
    F2K = np.full(n_frames, np.nan)
    H5K = np.full(n_frames, np.nan)

    # use DFT to find peak at +/- 50% of F0 at the given target
    # returns the amplitude of that peak and its frequency
    # intuition: this function takes a target frequency (2k or 5k) and says 'If I look at a specified interval around this target, what is the amplitude of the harmonic peak?'
    def dft_amplitude(segment, freq):
        n = np.arange(len(segment))
        v = np.exp(-1j * 2 * np.pi * freq * n / Fs)
        return 20 * np.log10(np.abs(segment @ v))
    def find_peak(segment, f_target, f0):
        f_min = f_target - 0.5 * f0
        f_max = f_target + 0.5 * f0
        result = minimize_scalar(
            lambda f: -dft_amplitude(segment, f),
            bounds=(f_min, f_max),
            method='bounded'
        )
        return -result.fun, result.x

    sampleshift = Fs / 1000

    # frame loop 
    for k in range(n_frames):
        f0 = F0[k]

        if np.isnan(f0) or f0 == 0:
            continue

        ks = round(k * sampleshift)
        N0 = Fs / f0

        ystart = round(ks - (n_periods / 2) * N0)
        yend   = round(ks + (n_periods / 2) * N0)

        if ystart < 0 or yend > len(y):
            continue

        if Fs / 2 <= 5000:
            continue

        segment = y[ystart:yend]

        # targets = 2000 and 5000
        # used in DFT peak finder 
        H2K[k], F2K[k] = find_peak(segment, 2000, f0) # include the frequency of the peak at 2k
        H5K[k], _      = find_peak(segment, 5000, f0)

    return H2K, F2K, H5K