import numpy as np
from scipy.signal import firwin, lfilter
from vspy.io import read_wav

# Low pass -> downsample
def _preprocess(y, fs, max_f0=500):
    dec = int(fs / (4 * max_f0))     # how much to downsample
    if dec <= 1:
        return y, fs, 1

    Fds = fs / dec

    # Low pass filter using 5ms hanning window
    # number of coefficients in the FIR filter (how long the filter is)
    n_taps = int(fs * 0.005) | 1
    # low pass filter
    b      = firwin(n_taps, Fds / 2, fs=fs, window='hann')
    # downsampled on the low pass filter
    y_pp   = lfilter(b, 1.0, y)[::dec]

    return y_pp, Fds, dec

# returns peaks after threshold filtering + parabolic interpolation
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
            # curvature around the three points {y0, y1, y2}
            a = (y2 - y1) + 0.5 * (y0 - y2)
            # don't calculate interpolation if the parabola is flat
            if abs(a) > 1e-6:
                # where the peak is (lag) 
                    # - b / 2a
                    # 2b = y2 - y0
                    # b = (y2-y0) / 2
                    # xp = -b/2a = (y0-y2) / 2 / 2a 
                xp = (y0 - y2) / (4.0 * a)
                # how strong the peak is (nccf value)
                yp = y1 - a * xp * xp
            # if curvature (a) = 0, then the offset stays at the integer peak 
            # return y1 for the NCCF value
            else:
                xp, yp = 0.0, y1
                # return lag and lag value
            candidates.append((k_min + i + xp, yp))
    return candidates

# remove direct current offset by subtracting the mean of the reference window from the entire frame
# prevents bias that inflates correlation at each lag
def _subtract_mean(frame, ref_size):
    mean = np.mean(frame[:ref_size])
    # frame = y_pp[p_ds : p_ds + n_ds + k_max_ds]
    # slice of waveform that covers the reference window and lags you want to test
    return frame - mean


# 2 pass NCCF call
def _get_candidates(y, y_pp, fs, Fds, dec, min_f0=40, max_f0=500, frame_step=0.01, wind_dur=0.0075, n_cands=20):
    # frame parameters at original sample rate (used in second pass)
    z     = round(frame_step * fs)   # frame step in samples
    n     = round(wind_dur * fs)     # window size in samples
    k_min = round(fs / max_f0)       # shortest lag you'll test 
    k_max = round(fs / min_f0)       # longest lag you'll test 
    K     = k_max - k_min + 1        # number of lags

    # frame parameters at downsampled rate (first NCCF pass)
    z_ds     = z // dec
    n_ds     = 1 + (n // dec)
    k_min_ds = max(1, k_min // dec)
    k_max_ds = k_max // dec
    K_ds     = 1 + (K // dec)

    # number of frames that fit in downsampled signal
    n_frames = (len(y_pp) - n_ds - k_max_ds) // z_ds

    all_candidates = []
    all_max_vals   = []

    # penalty that discourages long lags 
    lag_wt = 0.3 / K   # 

    for i in range(n_frames):
        # starting position for frame i in ds signal
        p_ds  = (i * z) // dec    
        frame_ds = _subtract_mean(y_pp[p_ds : p_ds + n_ds + k_max_ds], n_ds)
        ref   = frame_ds[:n_ds]
        # energy at reference window
        E_ref = np.dot(ref, ref)

        # coarse (first pass) NCCF on y_pp over all lags k_min_ds..k_max_ds
        nccf_ds = np.zeros(K_ds)
        if E_ref > 0:
            for ki in range(K_ds):
                k      = k_min_ds + ki
                lagged = frame_ds[k : k + n_ds]
                # energy of the lagged window 
                E_lag  = np.dot(lagged, lagged)
                if E_lag > 0:
                    nccf_ds[ki] = np.dot(ref, lagged) / np.sqrt(E_ref * E_lag)

        coarse_cands = _pick_peaks(nccf_ds, k_min_ds, cand_thresh=0.3)
        if len(coarse_cands) >= n_cands:
            # slicing to keep 1 fewer than n_cands
            # allows to save 1 spot for the voiceless hypothesis
            coarse_cands.sort(key=lambda x: -x[1])
            coarse_cands = coarse_cands[:n_cands - 1]

        # map to full sample rate and apply lag-dependent weight (larger lags = bad)
        coarse_cands = [(round(lag_ds * dec), yp * (1.0 - lag_wt * round(lag_ds * dec)))
                        for lag_ds, yp in coarse_cands]
        fine_lags = [lag for lag, _ in coarse_cands]

        # fine NCCF on y (non-pre-processed sample) in a 7-point vicinity around each candidate lag
        p        = i * z
        frame    = _subtract_mean(y[p : p + n + k_max], n)
        ref_fine = frame[:n]
        E_ref_fine = np.dot(ref_fine, ref_fine)

        fine_cands = []
        max_val    = 0.0
        if E_ref_fine > 0:
            for lag_full in fine_lags:
                # instead of searching over every lag, only search a 7-point window around each coarse candidate lag
                lag_lo = max(k_min, lag_full - 3)
                lag_hi = min(k_max, lag_full + 3)
                n_fine = lag_hi - lag_lo + 1
                # compute NCCF at full sample rate for those 7 lags around each coarse candidate
                nccf_fine = np.zeros(n_fine)
                for ki, k in enumerate(range(lag_lo, lag_hi + 1)):
                    lagged = frame[k : k + n]
                    E_lag = np.dot(lagged, lagged)
                    if E_lag > 0:
                        nccf_fine[ki] = np.dot(ref_fine, lagged) / np.sqrt(E_ref_fine * E_lag)
        
                max_val = max(max_val, nccf_fine.max())
                # store (integer_lag, peak, y0, y1, y2) — triplet used for backtracking interpolation
                if nccf_fine.max() > 0:
                    clip = 0.3 * nccf_fine.max()
                    for fi in range(1, len(nccf_fine) - 1):
                        if nccf_fine[fi] > clip and nccf_fine[fi] >= nccf_fine[fi-1] and nccf_fine[fi] >= nccf_fine[fi+1]:
                            fine_cands.append((lag_lo + fi, float(nccf_fine[fi]),
                                               float(nccf_fine[fi-1]), float(nccf_fine[fi]), float(nccf_fine[fi+1])))

        if len(fine_cands) >= n_cands:
            fine_cands.sort(key=lambda x: -x[1])
            fine_cands = fine_cands[:n_cands - 1]

        all_candidates.append(fine_cands)
        all_max_vals.append(max_val)

    return all_candidates, all_max_vals, n_frames, z, k_min, k_max


# extracts a fixed window from y (zero padding if window is out of bounds)
def _safe_extract(y, start, size):
    out = np.zeros(size)
    src_start = max(0, start)
    src_end   = min(len(y), start + size)
    if src_end > src_start:
        dst_start = src_start - start
        out[dst_start : dst_start + (src_end - src_start)] = y[src_start:src_end]
    return out


# computes how much two frames are changing in temrs of energy and spectral similairty
def _get_stationarity(y, fs, n_frames, z):
    stat_wsize = int(0.030 * fs)          # 30ms window
    stat_aint  = int(0.020 * fs)          # 20ms interval (gap between current and previous frame centers)
    ind        = (stat_aint - stat_wsize) // 2   # offset from frame start to window start

    stats      = []
    rms_ratios = []
# extract two frames and compute similarity
    for i in range(n_frames):
        curr_start = i * z + ind
        prev_start = curr_start - stat_aint
        curr_win   = _safe_extract(y, curr_start, stat_wsize)
        prev_win   = _safe_extract(y, prev_start, stat_wsize)
        stats.append(_spectral_similarity(curr_win, prev_win, fs))
        rms_ratios.append(_rms_ratio(curr_win, prev_win))

    return stats, rms_ratios


# assign local cost for each possible state for a given frame
# works on voiced and voiceless candiates
# each frame assigns has a number of candiates, and  adds a voiceless hypothesis
def _local_costs(fine_cands, max_val, k_max, lag_weight=0.3, voice_bias=0.0):
    lag_wt = lag_weight / k_max
    costs  = [1.0 - peak * (1.0 - lag * lag_wt) for lag, peak, *_ in fine_cands]
    costs.append(voice_bias + max_val)   # unvoiced hypothesis (for every frame, an unvoiced candiate with lag = -1 is added)
    return costs



# return energy ratio between frame i and frame i-1 (how much signal's loudness changed between 2 frames)
# take 1 frame, fit 30ms hanning window to it, calculate rms energy
# take previous frame, fit 30 ms hanning window, calculate rms energy 
# compute the ratio between the two
def _rms_ratio(curr_win, prev_win):
    n    = len(curr_win)
    hann = np.hanning(n)
    rms_curr = np.sqrt(np.dot(curr_win * hann, curr_win * hann) / n)
    rms_prev = np.sqrt(np.dot(prev_win * hann, prev_win * hann) / n)
    # handles edge cases 
    if rms_prev > 0.0:
        return (0.001 + rms_curr) / rms_prev
    elif rms_curr > 0.0:
        return 2.0   # energy increasing from silence
    else:
        return 1.0   # both silent

# return spectral similarity across frame i and frame i-1
# difficult to replicate. If there are issues with voice(less) to voice(less) transitions then you know its this
def _spectral_similarity(curr_win, prev_win, fs):
    order   = int(2 + fs / 1000)   # LPC order
    # differs from paper which calculates preemphasis based on sampling rate
    preemp  = 0.4                   # preemphasis

    def _preemph_hann(win):
        w      = win.astype(float).copy()
        w[1:] -= preemp * w[:-1]
        return w * np.hanning(len(w))

    def _lpc(frame, p):
        r = np.array([np.dot(frame[:len(frame)-k], frame[k:]) for k in range(p + 1)])
        if r[0] == 0:
            return np.zeros(p), r, 1.0
        R = np.array([[r[abs(i-j)] for j in range(p)] for i in range(p)])
        try:
            a = np.linalg.solve(R, -r[1:])
        except np.linalg.LinAlgError:
            return np.zeros(p), r, 1.0
        err = max(r[0] + np.dot(a, r[1:]), 1e-10)
        return a, r, err

    curr_w              = _preemph_hann(curr_win)
    prev_w              = _preemph_hann(prev_win)
    a_curr, _, _        = _lpc(curr_w, order)
    _,      r_prev, err_prev = _lpc(prev_w, order)

    # Itakura distance: how poorly the current frame's LPC predicts the previous frame
    a_full   = np.concatenate([[1.0], a_curr])
    itakura  = sum(a_full[i] * a_full[j] * r_prev[abs(i-j)]
                   for i in range(order + 1) for j in range(order + 1)) / err_prev
    itakura  = max(itakura, 0.81)   # clamp matching Snack

    return 0.2 / (itakura - 0.8)   # S = 0.2 / (itakura - 0.8)


# transition costs
def _transition_cost(lag1, lag2, stat, rms_ratio,
                     trans_cost=0.005, trans_amp=0.5, trans_spec=0.5,
                     freq_weight=0.02, double_cost=0.35, frame_step=0.01):
    # lag = -1 for unvoiced candidate 
    freq_wt = freq_weight / frame_step

    if lag1 == -1 and lag2 == -1:   # unvoiced -> unvoiced
        return 0.0

    if lag1 > 0 and lag2 > 0:       # voiced -> voiced
        ratio = np.log(lag2 / lag1)
        e_jk  = abs(ratio)
        # octave jumps get a reduced transition cost
        cost  = min(e_jk, double_cost + abs(ratio + np.log(2)))
        cost  = min(cost,  double_cost + abs(ratio - np.log(2)))
        return cost * freq_wt

    if lag1 == -1 and lag2 > 0:     # unvoiced -> voiced
        return trans_cost + trans_spec * stat + trans_amp / rms_ratio

    if lag1 > 0 and lag2 == -1:     # voiced -> unvoiced
        return trans_cost + trans_spec * stat + trans_amp * rms_ratio


# recursion step: finds the best path through the frames of an audio clip in terms of minimizing cost
def _dp_forward(all_candidates, all_max_vals, stats, rms_ratios, k_max,
                lag_weight=0.3, voice_bias=0.0, trans_cost=0.005,
                trans_amp=0.5, trans_spec=0.5, freq_weight=0.02,
                double_cost=0.35, frame_step=0.01):

    cum_costs = []
    backptrs  = []

    for i, (cands, max_val) in enumerate(zip(all_candidates, all_max_vals)):
        lags  = [c[0] for c in cands] + [-1]
        local = _local_costs(cands, max_val, k_max, lag_weight, voice_bias)

        if i == 0:
            cum_costs.append(local[:])
            backptrs.append([None] * len(lags))
            continue

        prev_lags = [c[0] for c in all_candidates[i - 1]] + [-1]
        prev_cum  = cum_costs[-1]

        # for each current state k, try every previous state j and find the one with the lowest cumulative cost plus transition cost
        frame_costs = []
        frame_bptrs = []
        for k, lag2 in enumerate(lags):
            totals = [prev_cum[j] + _transition_cost(lag1, lag2, stats[i], rms_ratios[i],
                      trans_cost, trans_amp, trans_spec, freq_weight, double_cost, frame_step)
                      for j, lag1 in enumerate(prev_lags)]
            best_j = int(np.argmin(totals))
            frame_costs.append(local[k] + totals[best_j])
            frame_bptrs.append(best_j)

        # stores a list of costs for frame i, one per state 
        cum_costs.append(frame_costs)
        # index j of the previous state that resulted in lowest cost for the current state k in frame i
        backptrs.append(frame_bptrs)

    return cum_costs, backptrs

# for each state in the best path, this function does parabolic interpolation to find the true peak, then calculates F0 
def _dp_backtrack(all_candidates, cum_costs, backptrs, fs):
    n_frames = len(cum_costs)
    k        = int(np.argmin(cum_costs[-1]))
    f0       = np.zeros(n_frames)

    for i in range(n_frames - 1, -1, -1):
        cands = all_candidates[i]
        if k < len(cands):
            lag_int, _, y0, y1, y2 = cands[k]
            # parabolic interpolation matching Snack's backtracking step
            den = 2.0 * (y0 + y2 - 2.0 * y1)
            lag = lag_int + (y0 - y2) / den if abs(den) > 1e-6 else lag_int
            f0[i] = fs / lag if lag > 0 else 0.0
        # else k == unvoiced: f0[i] stays 0
        if i > 0:
            k = backptrs[i][k]

    return f0


# final function 
def get_pitch_snack(wavfile, frameshift_ms=1, datalen=None, min_f0=40, max_f0=500,
                    wind_dur=0.025, n_cands=20, lag_weight=0.3, voice_bias=0.0,
                    trans_cost=0.005, trans_amp=0.5, trans_spec=0.5,
                    freq_weight=0.02, double_cost=0.35):
    # load audio
    y, fs = read_wav(wavfile)
    # convert frame shift to seconds
    frame_step = frameshift_ms / 1000.0
    # lowpass + downsample
    y_pp, Fds, dec = _preprocess(y, fs, max_f0)
    # 2 pass NCCF 
    all_candidates, all_max_vals, n_frames, z, _, k_max = _get_candidates(
        y, y_pp, fs, Fds, dec, min_f0, max_f0, frame_step, wind_dur, n_cands)
    n_voiced = sum(1 for c in all_candidates if len(c) > 0)

    # compute spectral similarity and energy ratios between adjacent frames 
    stats, rms_ratios = _get_stationarity(y, fs, n_frames, z)
    # compute costs and find best path 
    cum_costs, backptrs = _dp_forward(
        all_candidates, all_max_vals, stats, rms_ratios, k_max,
        lag_weight, voice_bias, trans_cost, trans_amp, trans_spec,
        freq_weight, double_cost, frame_step)
    # use backtracking function to find best peaks and calculate F0 
    raw_f0 = _dp_backtrack(all_candidates, cum_costs, backptrs, fs)
# create fixed-length output array
    if datalen is None:
        datalen = n_frames
    f0 = np.full(datalen, np.nan)
    for i, val in enumerate(raw_f0):
        t_ms = i * frameshift_ms
        idx  = round(t_ms / frameshift_ms)
        if 0 <= idx < datalen:
            f0[idx] = val if val > 0 else np.nan

    return f0
