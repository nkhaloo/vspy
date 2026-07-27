import numpy as np

from vspy.features._matlab import mround_int

# VoiceSauce vs_Initialize.m: vars.Nperiods_EC = 5.
NPERIODS_EC = 5


def get_energy(y, fs, f0, frame_shift=1.0, n_periods=NPERIODS_EC):

    y = np.asarray(y, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    sampleshift = fs / 1000.0 * frame_shift
    energy = np.full(len(f0), np.nan)

    for k in range(len(f0)):
        # VoiceSauce prints data(s) at t_ms = s (1-based MATLAB index) and vspy
        # row i is t_ms = i, so row k holds MATLAB iteration k.
        ks = mround_int(k * sampleshift)

        if ks <= 0 or ks > len(y):
            continue

        f0_curr = f0[k]
        if not np.isfinite(f0_curr) or f0_curr <= 0:
            continue

        N0 = fs / f0_curr  # period in samples

        # ystart/yend are 1-based sample indices (matching MATLAB)
        ystart = mround_int(ks - n_periods / 2 * N0)
        yend   = mround_int(ks + n_periods / 2 * N0) - 1

        if ystart <= 0 or yend > len(y):
            continue

        # convert to 0-based Python slice 
        yseg = y[ystart - 1 : yend]
        energy[k] = np.sum(yseg ** 2)

    return energy
