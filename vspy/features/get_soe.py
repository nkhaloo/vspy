import math
import numpy as np
import pandas as pd
from scipy.signal import lfilter, resample_poly

_ZFF_FS = 10_000  

# estimates a local moving-avergae trand and subracts it from the signal 
# centers oscillations of waveform around zero
def _remove_trend(x, N):
    N = int(round(N))
    if N <= 0:
        return x.copy()
    width = 2 * N + 1
    L = len(x)
    # moving average filter 
    c_causal = lfilter(np.ones(width) / width, 1.0, x)

    # special trend for edges of sample
    cb_raw = np.cumsum(x[: width - 2])         
    cbegin = cb_raw[::2] / np.arange(1, 2 * N, 2, dtype=float)  
    ce_raw = np.cumsum(x[L - 1 : L - 2 * N - 1 : -1]) 
    cend = ce_raw[-1::-2] / np.arange(2 * N - 1, 0, -2, dtype=float)  

    trend = np.concatenate([cbegin, c_causal[width - 1 :], cend])
    return x - trend

# zero frequency filter 
def _zff(x, n0):
    alpha = 0.999
    # shift waveform so that the first sample is 0
    x = x - x[0]
    # first pass filter
    s = lfilter([1.0, -1.0], [1.0], x)
    half_win = round(n0 / 1.5)
    u = lfilter([1.0], [1.0, -2 * alpha, alpha ** 2], s)
    v = _remove_trend(u, half_win)
    # second pass filter
    y = lfilter([1.0], [1.0, -2 * alpha, alpha ** 2], v)
    # final ZFF waveform
    z = _remove_trend(y, half_win)
    return z

# calculates local derivative of ZFF wavefom
# returns negative derivative at each downward 0 crossing 
def _htk_delta(x, D=2):

    N = len(x)
    dx = np.zeros(N)
    # computes derivative for samples near zero crossing on both sides and averages
    denom = 2.0 * sum(theta ** 2 for theta in range(1, D + 1))
    # which slope formula to use based on where nf is in the signal
    for nf in range(N):
        # middle of signal: look forward and backward
        if nf >= D and nf < N - D:
            numer = sum(theta * (x[nf + theta] - x[nf - theta])
                        for theta in range(1, D + 1))
            dx[nf] = numer / denom
        # beginning of signal: only look forward
        elif nf < D and nf + 1 < N:
            dx[nf] = x[nf + 1] - x[nf]
        # end of signal: only look backward
        elif nf >= N - D and nf > 0:
            dx[nf] = x[nf] - x[nf - 1]
    return dx


# main function
def get_soe(
    y: np.ndarray,
    fs: int,
    f0: np.ndarray,
    frameshift_ms: float = 1.0,
    method: str = "whole",
    window_ms: float = 25.0,
) -> pd.DataFrame:
    data_len = len(f0)
    f0 = np.asarray(f0, dtype=float)

  
    g = math.gcd(int(fs), _ZFF_FS)
    snd = resample_poly(y.astype(float), _ZFF_FS // g, int(fs) // g)
    Fs = _ZFF_FS

    if method == "whole":
        valid = f0[~np.isnan(f0)]
        if len(valid) == 0:
            return pd.DataFrame({"epoch": np.zeros(data_len, bool),
                                 "soe": np.full(data_len, np.nan)})
        mean_n0 = np.mean(Fs / valid)
        z = _zff(snd, mean_n0)

    elif method == "block":
        WINDOW = round(window_ms / 1000 * Fs)
        STEP = round(frameshift_ms / 1000 * Fs)
        nfrm = int((len(snd) + WINDOW) / Fs / (frameshift_ms / 1000))
        valid = f0[~np.isnan(f0)]
        fallback_f0 = float(np.mean(valid)) if len(valid) else 100.0
        z = np.zeros(len(snd))
        for nf in range(nfrm):
            ks = round(nf * STEP)
            if ks <= 0 or ks > len(snd):
                continue
            curr_f0 = f0[nf] if nf < data_len else np.nan
            if np.isnan(curr_f0):
                curr_f0 = fallback_f0
            curr_n0 = Fs / curr_f0
            win_start = round(ks - WINDOW / 2)
            if win_start <= 0 or win_start + WINDOW - 1 > len(snd):
                continue
            win_snd = snd[win_start - 1 : win_start - 1 + WINDOW] * np.hamming(WINDOW)
            curr_z = _zff(win_snd, curr_n0)
            z[win_start - 1 : win_start - 1 + WINDOW] += curr_z
    else:
        raise ValueError(f"method must be 'whole' or 'block', got {method!r}")
    z_max = np.max(np.abs(z))
    if z_max > 0:
        z = 0.95 * z / z_max
    z1 = np.concatenate([[np.nan], z[:-1]])
    tf = (z1 > 0) & (z <= 0)
    neg_deriv = -_htk_delta(z, D=2)

    epoch_out = np.zeros(data_len, dtype=bool)
    soe_out = np.full(data_len, np.nan)

    epoch_samples = np.where(tf)[0] 
    step_sec = frameshift_ms / 1000.0
    for smp_py in epoch_samples:
        t_smp = (smp_py + 1) / Fs
        frm_matlab = round(t_smp / step_sec)
        frm_py = frm_matlab - 1
        if 0 <= frm_py < data_len:
            epoch_out[frm_py] = True
            soe_out[frm_py] = neg_deriv[smp_py]

    return pd.DataFrame({"epoch": epoch_out, "soe": soe_out})
