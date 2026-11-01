
import numpy as np


def mround(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def mround_int(x):
    return int(mround(x))


def pickpeaks(y, winlen):
    y = np.asarray(y, dtype=float)
    empty = (np.array([], dtype=float), np.array([], dtype=int))
    if y.size < 2:
        return empty

    ymin = y.min()
    y = y - ymin


    dy = np.empty(y.size)
    dy[0] = y[0]
    dy[1:] = np.diff(y)

    inx1 = np.diff((dy < 0).astype(np.int8))
    if inx1.size == 0:
        return empty
    inx1[0] = 0
    inx1[-1] = 0

    inx = np.flatnonzero(inx1 > 0) + 1
    ymax = y[inx - 1].copy()

    nofmax = ymax.size
    if nofmax == 0:
        return empty
    if nofmax == 1:

        return ymax, inx


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
