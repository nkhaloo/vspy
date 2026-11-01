
import numpy as np
import soundfile as sf
import pandas as pd

from vspy.features._matlab import mround, mround_int


SHR_MIN = 40.0
SHR_MAX = 500.0
SHR_THRESHOLD = 0.4
CEILING = 1250.0
FRAME_PRECISION = 1



def _get_log_spectrum(segment, fftlen, limit_idx, logf, interp_logf):
    spec = np.fft.rfft(segment, n=fftlen)
    amp_full = np.abs(spec)

    amp = amp_full[1 : limit_idx + 2]
    interp_amp = np.interp(interp_logf, logf, amp)
    interp_amp = interp_amp - interp_amp.min()
    return interp_amp





def _twomax(x, lowerbound, upperbound, unitlen):
    region = x[lowerbound : upperbound + 1]
    if region.size == 0 or region.max() <= 0:
        return np.array([]), np.array([], dtype=int)

    idx1 = int(np.argmax(region)) + lowerbound
    mag1 = x[idx1]

    LIMIT = 0.0625
    harmonics = 2
    hi = min(len(x) - 1, upperbound)
    sp = idx1 + mround_int(np.log2(harmonics - LIMIT) / unitlen)
    if sp <= hi:
        ep = min(idx1 + mround_int(np.log2(harmonics + LIMIT) / unitlen), hi)
        region2 = x[sp : ep + 1]
        if region2.size and region2.max() > 0:
            idx2 = int(np.argmax(region2)) + sp
            return np.array([mag1, x[idx2]]), np.array([idx1, idx2])

    return np.array([mag1]), np.array([idx1])



def _compute_shr(log_spectrum, min_bin, N, shift_units, interp_len,
                 lowerbound, upperbound, shr_threshold):
    shshift = np.zeros((N, interp_len))
    shshift[0, :] = log_spectrum




    for i in range(2, N + 1):
        skip = mround_int(np.log2(i) / min_bin)
        if skip < interp_len:
            shshift[i - 1, : interp_len - skip] = log_spectrum[skip:]

    shsodd = shshift[0::2, :].sum(axis=0)
    shseven = shshift[1::2, :].sum(axis=0)
    difference = shseven - shsodd

    mags, indices = _twomax(difference, lowerbound, upperbound, min_bin)



    if mags.size == 0:
        return -1, 0.0
    if mags.size == 1:
        return int(indices[0]), 0.0

    shr = float((mags[0] - mags[1]) / (mags[0] + mags[1]))
    peak_index = int(indices[0] if shr > shr_threshold else indices[1])
    return peak_index, shr



def get_shrp(
    wavfile,
    frameshift_ms: float = 1.0,
    datalen=None,
    f0min: float = SHR_MIN,
    f0max: float = SHR_MAX,
    windowsize_ms: float = 25.0,
    shr_threshold: float = SHR_THRESHOLD,
    ceiling: float = CEILING,
) -> pd.DataFrame:
    y, fs = sf.read(str(wavfile), always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]

    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    interpolation_depth = 0.5

    N = int(np.floor(ceiling / f0min))
    N -= N % 2
    N *= 4

    segmentlen = mround_int(windowsize_ms * fs / 1000)
    inc = mround_int(frameshift_ms * fs / 1000)
    total_len = len(y)
    nf = (total_len - segmentlen + inc) // inc
    if nf <= 0:
        n = 0 if datalen is None else datalen
        return pd.DataFrame({"f0": np.full(n, np.nan), "shr": np.full(n, np.nan)})

    f0_time_ms = (np.arange(nf) * frameshift_ms) + windowsize_ms / 2.0

    fftlen = 1
    while fftlen < segmentlen * (1 + interpolation_depth):
        fftlen *= 2

    frequency = fs * np.arange(1, fftlen // 2 + 1) / fftlen
    limit_idx = int(np.where(frequency >= ceiling)[0][0])
    frequency = frequency[: limit_idx + 1]
    logf = np.log2(frequency)

    min_bin = logf[-1] - logf[-2]
    shift_units = mround_int(np.log2(N) / min_bin)

    n_steps = int(np.floor((logf[-1] - logf[0]) / min_bin))
    interp_logf = logf[0] + np.arange(n_steps + 1) * min_bin
    interp_len = len(interp_logf)

    maxlogf = np.log2(f0max / 2)
    minlogf = np.log2(f0min / 2)
    upper_hits = np.where(interp_logf >= maxlogf)[0]
    lower_hits = np.where(interp_logf >= minlogf)[0]
    upperbound = int(upper_hits[0]) if len(upper_hits) else interp_len - 1
    lowerbound = int(lower_hits[0]) if len(lower_hits) else 0

    newfre = 2.0 ** interp_logf
    hamming_win = np.hamming(segmentlen)

    curpos = mround(f0_time_ms / 1000 * fs).astype(int)

    f0_raw = np.zeros(nf)
    shr_raw = np.zeros(nf)
    cur_f0 = 0.0

    for n in range(nf):

        s = curpos[n] - mround_int(segmentlen / 2.0) - 1
        s = max(0, min(s, total_len - segmentlen))
        segment = y[s : s + segmentlen] * hamming_win

        log_spec = _get_log_spectrum(segment, fftlen, limit_idx, logf, interp_logf)

        peak_index, cur_shr = _compute_shr(
            log_spec, min_bin, N, shift_units, interp_len,
            lowerbound, upperbound, shr_threshold,
        )



        if peak_index != -1:
            cur_f0 = newfre[peak_index] * 2.0
            if cur_f0 > f0max:
                cur_f0 /= 2.0

        f0_raw[n] = cur_f0
        shr_raw[n] = cur_shr

    if datalen is None:
        datalen = nf

    f0_out = np.full(datalen, np.nan)
    shr_out = np.full(datalen, np.nan)





    t_rounded = mround(f0_time_ms).astype(int)
    finish = int(t_rounded[-1])
    step = int(frameshift_ms)
    for k_ms in range(0, finish + 1, step):
        diffs = np.abs(t_rounded - k_ms)
        inx = int(np.argmin(diffs))
        if diffs[inx] > FRAME_PRECISION * frameshift_ms:
            continue
        n = mround_int(k_ms / frameshift_ms) + 1
        if n < 1 or n > datalen:
            continue


        if n + 1 < datalen:
            f0_out[n + 1] = f0_raw[inx]
            shr_out[n + 1] = shr_raw[inx]

    return pd.DataFrame({"f0": f0_out, "shr": shr_out})
