# generate pitch using RAPT
# Same pitch detection as Snack audio toolkit 

import numpy as np
from scipy.signal import lfilter
from vspy.io import read_wav

# define helper function for downsampling 
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

# define normalized cross correlation function
# frame must be at least window_size + lag_max samples long
# ref window is always the first window_size samples; lagged window shifts by lag
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

# define a function that picks probably candidates from the CCF call 
# based on a threshold and a local maximum test (maxima have to b ehigher than local neighbors)
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

# define a function that denotes how spectrally stable a candidate is
# itakura distanceː LPC fits a curve estimating spectral envelope. Itakura distance measures how badly a model fitted to frame A fails when used to predict frame B
# helps with a transition from a voiced (periodic) to voiceless sound 
def _lpc_stationarity(prev_frame, curr_frame, order):
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

    a_curr, _        = lpc(curr_frame, order)
    _,      err_prev = lpc(prev_frame, order)

    r_prev = np.array([np.dot(prev_frame[:len(prev_frame)-k], prev_frame[k:])
                       for k in range(order+1)])
    if r_prev[0] == 0:
        return 0.01 * 0.2

    a_curr_full = np.concatenate([[1.0], a_curr])
    itakura = 0.0
    for i in range(order + 1):
        for j in range(order + 1):
            itakura += a_curr_full[i] * a_curr_full[j] * r_prev[abs(i-j)]
    itakura /= err_prev
    itakura = max(itakura, 0.81)

    return 0.2 / (itakura - 0.8)


# define forward passː goes through each frame and computes transition costs
# a huge difference in F0 has a high transition cost 
# this part also denotes when the frame is likely voiceless 
def _dp_forward(all_candidates, stationarity, fs, params):
    lag_weight  = params['lag_weight']
    freq_weight = params['freq_weight']
    trans_cost  = params['trans_cost']
    trans_amp   = params['trans_amp']
    trans_spec  = params['trans_spec']
    # can tune: positive value = bias towards voice. negative = bias towards voiceless
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
        # max_val is going to be high at high a periodic peak 
        # low peak means smaller max_val and likely unvoiced
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
                    trans = trans_cost + trans_spec * stationarity[i] + trans_amp
                elif k == n and j < n_prev:
                    trans = trans_cost + trans_spec * stationarity[i] + trans_amp
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

# define backtrackingː pick candidates with the lowest accumulated cost
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



# define the main function
def get_pitch_snack(wavfile, frameshift_ms=1, datalen=None, min_f0=60, max_f0=400):
    y, fs = read_wav(wavfile)
    # set hyperparamters
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

    all_candidates  = []
    stationarity    = []
    prev_stat_frame = np.zeros(frame_size)

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
            fine_min  = max(lag_min, lag_full - 2)
            fine_max  = min(lag_max, lag_full + 2)
            fine_ccf  = _normalized_ccf(frame, fine_min, fine_max,
                                         window_size=frame_size)
            fine_peak = int(np.argmax(fine_ccf))
            fine_lag  = fine_min + fine_peak
            fine_val  = fine_ccf[fine_peak]
            fine_cands.append((fine_lag, fine_val))

        fine_cands.sort(key=lambda x: -x[1])
        all_candidates.append(fine_cands)

        stat = _lpc_stationarity(prev_stat_frame, frame, stat_order)
        stationarity.append(stat)
        prev_stat_frame = frame

    params = {
        'lag_weight':  0.3,
        'freq_weight': 0.02,
        'trans_cost':  0.005,
        'trans_amp':   0.5,
        'trans_spec':  0.5,
        'voice_bias':  0.0,
        'double_cost': 0.35,
        'lag_max':     lag_max,
    }

    costs, backptrs = _dp_forward(all_candidates, stationarity, fs, params)
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
