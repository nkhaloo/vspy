import numpy as np


def get_energy(y, fs, f0, frame_shift=1.0, n_periods=5):

    f0 = np.asarray(f0, dtype=float)
    sampleshift = fs / 1000.0 * frame_shift
    energy = np.full(len(f0), np.nan)

    for k in range(len(f0)):
        # MATLAB is 1-indexed: frame k (1-based) -> ks = round(k * sampleshift)
        # Python k is 0-indexed, so the equivalent MATLAB k is k+1
        ks = round((k + 1) * sampleshift)

        if ks <= 0 or ks > len(y):
            continue

        f0_curr = f0[k]
        if np.isnan(f0_curr) or f0_curr == 0:
            continue

        N0 = fs / f0_curr  # period in samples

        # ystart/yend are 1-based sample indices (matching MATLAB)
        ystart = round(ks - n_periods / 2 * N0)
        yend   = round(ks + n_periods / 2 * N0) - 1

        if ystart <= 0 or yend > len(y):
            continue

        # convert to 0-based Python slice 
        yseg = y[ystart - 1 : yend]
        energy[k] = np.sum(yseg ** 2)

    return energy
