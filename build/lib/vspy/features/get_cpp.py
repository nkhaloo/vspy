
import numpy as np

from vspy.features._matlab import mround_int, pickpeaks


NPERIODS_EC = 5


def get_cpp(y, fs, f0, n_periods=NPERIODS_EC, frame_shift=1.0):
    y = np.asarray(y, dtype=float)
    f0 = np.asarray(f0, dtype=float)

    sample_shift = fs / 1000.0 * frame_shift

    n_ms = mround_int(fs / 1000.0)

    cpp = np.full(len(f0), np.nan)

    for k in range(len(f0)):
        ks = mround_int(k * sample_shift)
        if ks <= 0 or ks > len(y):
            continue

        f0_curr = f0[k]


        if not np.isfinite(f0_curr) or f0_curr <= 0:
            continue

        n0_curr = fs / f0_curr

        ystart = mround_int(ks - n_periods / 2 * n0_curr)
        yend = mround_int(ks + n_periods / 2 * n0_curr) - 1
        if ystart <= 0 or yend > len(y):
            continue

        yseg = y[ystart - 1:yend]
        yseg = yseg * np.hamming(len(yseg))

        with np.errstate(divide="ignore", invalid="ignore"):
            yseg_c = np.fft.ifft(np.log(np.abs(np.fft.fft(yseg)))).real
            yseg_c_db = 10 * np.log10(yseg_c ** 2)

        yseg_c_db = yseg_c_db[:len(yseg) // 2]
        if n_ms >= len(yseg_c_db):
            continue



        region = yseg_c_db[n_ms - 1:]
        if not np.all(np.isfinite(region)):
            continue

        _, inx = pickpeaks(region, 2 * n0_curr)
        if inx.size == 0:
            continue



        pinx = int(np.argmin(np.abs(inx - n0_curr)))
        peak_1b = int(inx[pinx]) + n_ms - 1
        peak_val = yseg_c_db[peak_1b - 1]

        x = np.arange(n_ms, len(yseg_c_db) + 1, dtype=float)
        p = np.polyfit(x, region, 1)
        base_val = p[0] * peak_1b + p[1]

        cpp[k] = peak_val - base_val

    return cpp
