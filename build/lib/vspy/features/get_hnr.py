
import numpy as np

from vspy.features._matlab import mround_int


NPERIODS_EC = 5


def get_hnr(y, Fs, F0, n_periods=NPERIODS_EC, frameshift_ms=1.0):
    y = np.asarray(y, dtype=float)
    F0 = np.asarray(F0, dtype=float)

    n_frames = len(F0)
    sampleshift = Fs / 1000.0 * frameshift_ms

    HNR05 = np.full(n_frames, np.nan)
    HNR15 = np.full(n_frames, np.nan)
    HNR25 = np.full(n_frames, np.nan)
    HNR35 = np.full(n_frames, np.nan)

    for k in range(n_frames):
        ks = mround_int(k * sampleshift)
        if ks <= 0 or ks > len(y):
            continue

        f0_curr = F0[k]
        if not np.isfinite(f0_curr) or f0_curr <= 0:
            continue

        N0_curr = 1.0 / f0_curr * Fs

        ystart = mround_int(ks - n_periods / 2 * N0_curr)
        yend = mround_int(ks + n_periods / 2 * N0_curr) - 1
        if (yend - ystart + 1) % 2 == 0:
            yend -= 1

        if ystart <= 0 or yend > len(y):
            continue

        hnr = _hnr_segment(y[ystart - 1:yend], f0_curr, Fs)

        HNR05[k] = hnr[0]
        HNR15[k] = hnr[1]
        HNR25[k] = hnr[2]
        HNR35[k] = hnr[3]

    return HNR05, HNR15, HNR25, HNR35


def _hnr_segment(y, F0, Fs, n_freqs=(500, 1500, 2500, 3500)):
    NBins = len(y)
    N0 = mround_int(Fs / F0)
    N0_delta = mround_int(N0 * 0.1)

    y = y * np.hamming(NBins)
    Y = np.fft.fft(y, NBins)
    with np.errstate(divide="ignore"):
        aY = np.log10(np.abs(Y))
    ay = np.fft.ifft(aY).real


    n_rahmonics = NBins // 2 // N0 if N0 > 0 else 0
    for k in range(1, n_rahmonics + 1):
        seg_start = k * N0 - N0_delta - 1
        seg_end = k * N0 + N0_delta
        if seg_start < 0 or seg_end > len(ay):
            continue
        ayseg = ay[seg_start:seg_end]
        if len(ayseg) < 2:
            continue

        inx = int(np.argmax(np.abs(ayseg)))
        s_ayseg = np.sign(np.diff(ayseg))





        left = np.flatnonzero(s_ayseg[inx - 1::-1] != 1) if inx > 0 else np.array([], dtype=int)
        right = np.flatnonzero(s_ayseg[inx + 1:] == 1)
        if left.size == 0 or right.size == 0:
            continue

        l_inx = inx - int(left[0])
        r_inx = inx + int(right[0]) + 1
        ay[seg_start + l_inx:seg_start + r_inx + 1] = 0.0


    midL = mround_int(NBins / 2.0) + 1
    ay[midL - 1:] = ay[midL - 2:midL - 2 - (len(ay) - midL) - 1:-1]

    Nap = np.fft.fft(ay).real
    N_spec = Nap.copy()
    Ha = aY - Nap


    Hdelta = F0 / Fs * NBins
    stop = mround_int(NBins / 2.0)
    i = 1
    while True:
        f = Hdelta * i + 0.0001
        if f > stop:
            break
        fstart = int(np.ceil(f - Hdelta))
        fend = mround_int(f)
        if fstart <= fend:
            seg = Ha[fstart - 1:fend]
            if seg.size:
                N_spec[fstart - 1:fend] -= abs(float(np.min(seg)))
        i += 1


    H_spec = aY - N_spec
    hnr = np.zeros(len(n_freqs))
    for j, freq in enumerate(n_freqs):
        Ef = mround_int(freq / Fs * NBins)
        if Ef > 0:
            hnr[j] = 20 * np.mean(H_spec[:Ef]) - 20 * np.mean(N_spec[:Ef])

    return hnr
