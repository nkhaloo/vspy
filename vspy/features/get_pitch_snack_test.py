


import numpy as np
from scipy.signal import lfilter
from vspy.io import read_wav



def _downsample(y, fs, target_fs=2000):
    decimate = int(fs / target_fs)
    if decimate <= 1:
        return y, fs
    n_coeff = int(fs * 0.005) | 1
    cutoff = 0.5 / decimate
    t = np.arange(n_coeff)
    mid = n_coeff // 2
    h = np.sinc(2 * cutoff * (t - mid))
    h *= 0.5 - 0.5 * np.cos(2 * np.pi * (t + 0.5) / n_coeff)
    h /= h.sum()
    filtered = lfilter(h, 1.0, y)
    return filtered[::decimate], fs // decimate



def _normalized_ccf(frame, lag_min, lag_max, window_size=None):
    if window_size is None:
        window_size = len(frame) - lag_max
    if window_size <= 0:
        return np.zeros(lag_max - lag_min + 1)

    ref = frame[:window_size]
    energy_ref = np.dot(ref, ref)
    if energy_ref == 0:
        return np.zeros(lag_max - lag_min + 1)
    ccf = np.zeros(lag_max - lag_min + 1)
    for i, lag in enumerate(range(lag_min, lag_max + 1)):

        b = frame[lag : lag + window_size]
        if len(b) < window_size:
            ccf[i] = 0.0
            continue

        energy_lagged = np.dot(b, b)
        if energy_lagged == 0:
            ccf[i] = 0.0


        else:
            ccf[i] = np.dot(ref, b) / np.sqrt(energy_ref * energy_lagged)

    return ccf



def _get_candidates(ccf, lag_min, cand_thresh, n_cands=20):
    candidates = []
    peak = np.max(ccf)
    if peak == 0:
        return candidates
    clip = cand_thresh * peak


    for i in range(1, len(ccf) - 1):
        if ccf[i] > clip and ccf[i] >= ccf[i-1] and ccf[i] >= ccf[i+1]:
            lag = lag_min + i
            candidates.append((lag, ccf[i]))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n_cands - 1]



def _parabolic_peak(ccf, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(ccf) - 1:
        return float(peak_idx), float(ccf[peak_idx])
    y0, y1, y2 = ccf[peak_idx - 1], ccf[peak_idx], ccf[peak_idx + 1]
    a = (y2 - y1) + 0.5 * (y0 - y2)
    if abs(a) > 1e-6:
        xp = (y0 - y2) / (4.0 * a)
        yp = y1 - a * xp * xp
    else:
        xp = 0.0
        yp = y1
    return peak_idx + xp, yp


def _safe_extract(y, start, size):
    out = np.zeros(size)
    src_start = max(0, start)
    src_end   = min(len(y), start + size)
    if src_end > src_start:
        dst_start = src_start - start
        out[dst_start : dst_start + (src_end - src_start)] = y[src_start:src_end]
    return out










def _get_similarity(prev_win, curr_win, order, preemp=0.4):
    def _preemph_hann(win):
        w = win.astype(float)

        w[1:] = w[1:] - preemp * w[:-1]

        n = len(w)
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)
        return w * hann

    def lpc(frame, p):
        r = np.array([np.dot(frame[:len(frame)-k], frame[k:]) for k in range(p+1)])
        if r[0] == 0:
            return np.zeros(p), 1.0
        R = np.array([[r[abs(i-j)] for j in range(p)] for i in range(p)])
        try:
            a = np.linalg.solve(R, -r[1:])
        except np.linalg.LinAlgError:
            return np.zeros(p), 1.0
        err = r[0] + np.dot(a, r[1:])
        return a, max(err, 1e-10)

    curr_w = _preemph_hann(curr_win)
    prev_w = _preemph_hann(prev_win)


    rms_curr = np.sqrt(np.dot(curr_w, curr_w) / len(curr_w)) if len(curr_w) > 0 else 0.0
    rms_prev = np.sqrt(np.dot(prev_w, prev_w) / len(prev_w)) if len(prev_w) > 0 else 0.0

    a_curr, _        = lpc(curr_w, order)
    _,      err_prev = lpc(prev_w, order)



    r_prev = np.array([np.dot(prev_w[:len(prev_w)-k], prev_w[k:])
                       for k in range(order+1)])
    if r_prev[0] == 0:
        stat = 0.01 * 0.2
    else:
        a_curr_full = np.concatenate([[1.0], a_curr])

        itakura = 0.0
        for i in range(order + 1):
            for j in range(order + 1):
                itakura += a_curr_full[i] * a_curr_full[j] * r_prev[abs(i-j)]
        itakura /= err_prev
        itakura = max(itakura, 0.81)
        stat = 0.2 / (itakura - 0.8)


    if rms_prev > 0.0:
        rms_ratio = (0.001 + rms_curr) / rms_prev
    elif rms_curr > 0.0:
        rms_ratio = 2.0
    else:
        rms_ratio = 1.0

    return stat, rms_ratio





def _dp_forward(all_candidates, stationarity, rms_ratios, params):
    lag_weight  = params['lag_weight']
    freq_weight = params['freq_weight']
    trans_cost  = params['trans_cost']
    trans_amp   = params['trans_amp']
    trans_spec  = params['trans_spec']

    voice_bias  = params['voice_bias']
    double_cost = params['double_cost']
    lag_max     = params['lag_max']

    costs     = []
    backptrs  = []

    for i, cands in enumerate(all_candidates):
        n = len(cands)
        local_costs = np.zeros(n + 1)
        for k, (lag, val) in enumerate(cands):
            local_costs[k] = 1.0 - val * (1.0 - lag * lag_weight / lag_max)
        max_val = max((v for _, v in cands), default=0.0)


        local_costs[n] = voice_bias + max_val

        if i == 0:
            costs.append(local_costs.copy())
            backptrs.append(np.zeros(n + 1, dtype=int))
            continue

        prev_cands = all_candidates[i - 1]
        prev_costs = costs[i - 1]
        n_prev = len(prev_cands)
        new_costs = np.zeros(n + 1)
        new_back  = np.zeros(n + 1, dtype=int)

        for k in range(n + 1):
            best_cost = np.inf
            best_prev = 0
            for j in range(n_prev + 1):
                if k < n and j < n_prev:
                    loc2 = cands[k][0]
                    loc1 = prev_cands[j][0]
                    ratio = np.log(loc2 / loc1)
                    t = abs(ratio)
                    t = min(t, double_cost + abs(ratio + np.log(2)))
                    t = min(t, double_cost + abs(ratio - np.log(2)))
                    trans = t * freq_weight
                elif k < n and j == n_prev:

                    trans = trans_cost + trans_spec * stationarity[i] + trans_amp / rms_ratios[i]
                elif k == n and j < n_prev:

                    trans = trans_cost + trans_spec * stationarity[i] + trans_amp * rms_ratios[i]
                else:
                    trans = 0.0
                total = prev_costs[j] + trans
                if total < best_cost:
                    best_cost = total
                    best_prev = j
            new_costs[k] = local_costs[k] + best_cost
            new_back[k]  = best_prev

        costs.append(new_costs)
        backptrs.append(new_back)

    return costs, backptrs


def _dp_backtrack(all_candidates, costs, backptrs):
    n_frames = len(costs)
    best_lags = np.zeros(n_frames)

    k = int(np.argmin(costs[-1]))
    for i in range(n_frames - 1, -1, -1):
        cands = all_candidates[i]
        if k < len(cands):
            best_lags[i] = cands[k][0]
        else:
            best_lags[i] = 0
        k = backptrs[i][k]

    return best_lags




def get_pitch_snack(wavfile, frameshift_ms=1, datalen=None, min_f0=60, max_f0=400):
    y, fs = read_wav(wavfile)

    frame_step  = int(round(frameshift_ms / 1000 * fs))
    frame_size  = int(round(0.0075 * fs))
    lag_min     = int(round(fs / max_f0))
    lag_max     = int(round(fs / min_f0))
    y_ds, fs_ds = _downsample(y, fs)
    lag_min_ds  = max(1, int(round(fs_ds / max_f0)))
    lag_max_ds  = int(round(fs_ds / min_f0))

    stat_order   = int(2 + fs / 1000)
    ncomp        = frame_size + lag_max + 1
    win_ds       = int(round(frame_size * fs_ds / fs))
    ncomp_ds     = win_ds + lag_max_ds + 1
    n_frames     = (len(y) - ncomp) // frame_step



    stat_wsize   = int(round(0.030 * fs))
    stat_aint    = int(round(0.020 * fs))
    stat_ind     = (stat_aint - stat_wsize) // 2

    all_candidates  = []
    stationarity    = []
    rms_ratios      = []

    for i in range(n_frames):
        start = i * frame_step
        frame    = y[start : start + ncomp]

        ds_start = int(round(start * fs_ds / fs))
        ds_frame = y_ds[ds_start : ds_start + ncomp_ds]

        coarse_ccf   = _normalized_ccf(ds_frame, lag_min_ds, lag_max_ds,
                                        window_size=win_ds)
        coarse_cands = _get_candidates(coarse_ccf, lag_min_ds, cand_thresh=0.3)

        fine_cands = []
        for lag_ds, _ in coarse_cands:
            lag_full  = int(round(lag_ds * fs / fs_ds))
            fine_min  = max(lag_min, lag_full - 3)
            fine_max  = min(lag_max, lag_full + 3)
            fine_ccf  = _normalized_ccf(frame, fine_min, fine_max,
                                         window_size=frame_size)

            fine_peak_idx      = int(np.argmax(fine_ccf))
            fine_xp, fine_yp   = _parabolic_peak(fine_ccf, fine_peak_idx)
            fine_cands.append((fine_min + fine_xp, fine_yp))

        fine_cands.sort(key=lambda x: -x[1])
        all_candidates.append(fine_cands)



        curr_stat_start = start + stat_ind
        prev_stat_start = curr_stat_start - stat_aint
        curr_stat_win   = _safe_extract(y, curr_stat_start, stat_wsize)
        prev_stat_win   = _safe_extract(y, prev_stat_start, stat_wsize)

        stat, rms_ratio = _get_similarity(prev_stat_win, curr_stat_win, stat_order)
        stationarity.append(stat)
        rms_ratios.append(rms_ratio)

    params = {
        'lag_weight':  0.3,

        'freq_weight': 0.02 * fs / frame_step,
        'trans_cost':  0.005,
        'trans_amp':   0.5,
        'trans_spec':  0.5,
        'voice_bias':  0.0,
        'double_cost': 0.35,
        'lag_max':     lag_max,
    }

    costs, backptrs = _dp_forward(all_candidates, stationarity, rms_ratios, params)
    best_lags = _dp_backtrack(all_candidates, costs, backptrs)

    if datalen is None:
        datalen = n_frames

    f0 = np.full(datalen, np.nan)
    for i, lag in enumerate(best_lags):
        t_ms = i * frame_step / fs * 1000
        idx  = round(t_ms / frameshift_ms)
        if 0 <= idx < datalen and lag > 0:
            f0[idx] = fs / lag

    return f0
