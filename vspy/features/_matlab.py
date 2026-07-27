"""Small helpers that reproduce MATLAB semantics exactly.

VoiceSauce is a MATLAB program, so a faithful port has to match MATLAB's
rounding rule and its peak picker.  Both are used by several feature modules,
so they live here rather than being duplicated.
"""

import numpy as np


def mround(x):
    """MATLAB `round`: half away from zero (numpy/Python round half to even)."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def mround_int(x):
    """Scalar `mround` returned as a Python int."""
    return int(mround(x))


def pickpeaks(y, winlen):
    """Port of VoiceSauce func_pickpeaks.m.

    Returns (ymax, inx) where `inx` holds **1-based** peak positions, exactly as
    the MATLAB function does -- callers in VoiceSauce use those 1-based values
    directly in arithmetic, so the convention has to be preserved.
    """
    y = np.asarray(y, dtype=float)
    empty = (np.array([], dtype=float), np.array([], dtype=int))
    if y.size < 2:
        return empty

    ymin = y.min()
    y = y - ymin

    # dy = [y(1); diff(y)] -- diff() shifts by one, so one sample is inserted
    dy = np.empty(y.size)
    dy[0] = y[0]
    dy[1:] = np.diff(y)

    inx1 = np.diff((dy < 0).astype(np.int8))  # +1 for maxima, -1 for minima
    if inx1.size == 0:
        return empty
    inx1[0] = 0    # do not accept maxima at beginning
    inx1[-1] = 0   # do not accept maxima at end

    inx = np.flatnonzero(inx1 > 0) + 1  # 1-based positions
    ymax = y[inx - 1].copy()

    nofmax = ymax.size
    if nofmax == 0:
        return empty
    if nofmax == 1:
        # MATLAB returns here *before* adding ymin back -- replicated as-is.
        return ymax, inx

    # filter maxima with a window of length winlen
    for cnt in range(1, nofmax + 1):
        cmp_ = inx[cnt - 1] - winlen
        sel = np.flatnonzero(inx[:cnt] > cmp_)
        if sel.size == 0:
            continue
        sub = ymax[sel]
        mi = int(np.argmax(sub))
        ymax[sel] = -60000.0
        ymax[sel[0] + mi] = sub[mi]

    keep = np.flatnonzero(ymax > 0)
    return ymax[keep] + ymin, inx[keep]
