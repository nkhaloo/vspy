import numpy as np
from scipy.signal import firwin, lfilter
from vspy.io import read_wav

def _preprocess(y, fs, max_f0=500):
    dec = int(fs / (4 * max_f0))     # floor, matching Snack's (int) cast
    if dec <= 1:
        return y, fs, 1

    Fds = fs / dec

    # FIR lowpass: Hanning-windowed sinc, 5ms filter length
    n_taps = int(fs * 0.005) | 1
    # low pass filter
    b      = firwin(n_taps, Fds / 2, fs=fs, window='hann')
    # downsampled on the low pass filter
    y_pp   = lfilter(b, 1.0, y)[::dec]

    return y_pp, Fds, dec


def _pick_peaks(nccf, k_min, cand_thresh):
    max_val = nccf.max()
    if max_val <= 0:
        return []
    clip = cand_thresh * max_val
    candidates = []
    for i in range(1, len(nccf) - 1):
        if nccf[i] > clip and nccf[i] >= nccf[i-1] and nccf[i] >= nccf[i+1]:
            # parabolic interpolation to refine peak location and value
            y0, y1, y2 = nccf[i-1], nccf[i], nccf[i+1]
            a = (y2 - y1) + 0.5 * (y0 - y2)
            if abs(a) > 1e-6:
                xp = (y0 - y2) / (4.0 * a)
                yp = y1 - a * xp * xp
            else:
                xp, yp = 0.0, y1
            candidates.append((k_min + i + xp, yp))
    return candidates


def _get_candidates(y, y_pp, fs, Fds, dec, min_f0=40, max_f0=500, frame_step=0.01, wind_dur=0.0075, n_cands=20):
    # frame parameters at original sample rate
    z     = round(frame_step * fs)   # frame step in samples
    n     = round(wind_dur * fs)     # window size in samples
    k_min = round(fs / max_f0)       # minimum lag
    k_max = round(fs / min_f0)       # maximum lag
    K     = k_max - k_min + 1        # number of lags

    # frame parameters at downsampled rate (coarse NCCF pass)
    z_ds     = z // dec
    n_ds     = 1 + (n // dec)
    k_min_ds = max(1, k_min // dec)
    k_max_ds = k_max // dec
    K_ds     = 1 + (K // dec)

    n_frames = (len(y_pp) - n_ds - k_max_ds) // z_ds

    all_candidates = []

    lag_wt = 0.3 / K   # lag_weight / nlags, matching Snack's lag_wt = par->lag_weight/nlags

    for i in range(n_frames):
        p_ds = (i * z) // dec    # matching Snack's decind = (ind * step)/dec
        ref  = y_pp[p_ds : p_ds + n_ds]
        E_ref = np.dot(ref, ref)

        # coarse (first pass) NCCF on y_pp over all lags k_min_ds..k_max_ds
        nccf_ds = np.zeros(K_ds)
        if E_ref > 0:
            for ki in range(K_ds):
                k      = k_min_ds + ki
                lagged = y_pp[p_ds + k : p_ds + k + n_ds]
                E_lag  = np.dot(lagged, lagged)
                if E_lag > 0:
                    nccf_ds[ki] = np.dot(ref, lagged) / np.sqrt(E_ref * E_lag)

        coarse_cands = _pick_peaks(nccf_ds, k_min_ds, cand_thresh=0.3)
        if len(coarse_cands) >= n_cands:
            coarse_cands.sort(key=lambda x: -x[1])
            coarse_cands = coarse_cands[:n_cands - 1]

        # map to full sample rate and apply lag-dependent weight
        # matching Snack: *lp = (*lp * dec) + (int)(0.5+(xp*dec))
        #                 *pe = yp*(1.0f - (lag_wt * *lp))
        coarse_cands = [(round(lag_ds * dec), yp * (1.0 - lag_wt * round(lag_ds * dec)))
                        for lag_ds, yp in coarse_cands]
        fine_lags = [lag for lag, _ in coarse_cands]

        # fine NCCF on y in a 7-point vicinity around each candidate lag
        p = i * z
        ref_fine   = y[p : p + n]
        E_ref_fine = np.dot(ref_fine, ref_fine)

        fine_cands = []
        if E_ref_fine > 0:
            for lag_full in fine_lags:
                lag_lo = max(k_min, lag_full - 3)
                lag_hi = min(k_max, lag_full + 3)
                n_fine = lag_hi - lag_lo + 1

                nccf_fine = np.zeros(n_fine)
                for ki, k in enumerate(range(lag_lo, lag_hi + 1)):
                    lagged = y[p + k : p + k + n]
                    if len(lagged) < n:
                        continue
                    E_lag = np.dot(lagged, lagged)
                    if E_lag > 0:
                        nccf_fine[ki] = np.dot(ref_fine, lagged) / np.sqrt(E_ref_fine * E_lag)

                fine_cands.extend(_pick_peaks(nccf_fine, lag_lo, cand_thresh=0.3))

        if len(fine_cands) >= n_cands:
            fine_cands.sort(key=lambda x: -x[1])
            fine_cands = fine_cands[:n_cands - 1]

        all_candidates.append(fine_cands)

    return all_candidates, n_frames, z, k_min, k_max


# stationaryː LPC similarity
# DP forward pass (accumulate cost)
# DP backprop (find best path)