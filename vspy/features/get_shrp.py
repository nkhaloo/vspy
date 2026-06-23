import numpy as np
import soundfile as sf
import pandas as pd
from scipy.interpolate import interp1d as scipy_interp1d

# take the spectrum for 1 frame and return log_spec vector
def _get_log_spectrum(segment, fftlen, limit_idx, logf, interp_logf):
    spec = np.fft.rfft(segment, n=fftlen)
    amp_full = np.abs(spec)
    amp = amp_full[1 : limit_idx + 2]
    f = scipy_interp1d(logf, amp, kind="linear", bounds_error=False, fill_value=0.0)
    interp_amp = f(interp_logf)
    interp_amp -= interp_amp.min()
    return interp_amp

# takes full difference between even and odd rows for 1 frame and returns: 
# the strongest possible peak in the allowed F0 range 
# a competing peak at double that frequency
def _twomax(x, lowerbound, upperbound, unitlen):
    region = x[lowerbound : upperbound + 1]
    if region.max() <= 0:
        return np.array([]), np.array([], dtype=int)

    local_idx = int(np.argmax(region))
    idx1 = local_idx + lowerbound
    mag1 = x[idx1]

    LIMIT = 0.0625
    harmonics = 2
    sp = idx1 + round(np.log2(harmonics - LIMIT) / unitlen)
    ep = idx1 + round(np.log2(harmonics + LIMIT) / unitlen)
    ep = min(ep, min(len(x) - 1, upperbound))

    if sp <= min(len(x) - 1, upperbound) and sp <= ep:
        region2 = x[sp : ep + 1]
        if len(region2) > 0 and region2.max() > 0:
            idx2 = int(np.argmax(region2)) + sp
            mag2 = x[idx2]
            return np.array([mag1, mag2]), np.array([idx1, idx2])

    return np.array([mag1]), np.array([idx1])

# computes distance matrix across all candidate F0, returns peaks, calculates SHR
def _compute_shr(log_spectrum, min_bin, N, shift_units, interp_len,
                 lowerbound, upperbound, shr_threshold):
    shshift = np.zeros((N, interp_len))
    shshift[0, :] = log_spectrum

    for i_py in range(1, N):
        skip = round(np.log2(i_py + 1) / min_bin)
        if skip < interp_len:
            shshift[i_py, : interp_len - skip] = log_spectrum[skip:]

    shsodd  = shshift[0::2, :].sum(axis=0)
    shseven = shshift[1::2, :].sum(axis=0)
    difference = shseven - shsodd

    mags, indices = _twomax(difference, lowerbound, upperbound, min_bin)

    if len(mags) == 0 or mags[0] <= 0:
        return -1, 0.0

    if len(mags) == 1:
        return int(indices[0]), 0.0

    shr = float((mags[0] - mags[1]) / (mags[0] + mags[1]))
    peak_index = int(indices[0] if shr > shr_threshold else indices[1])
    return peak_index, shr

# main function
def get_shrp(
    wavfile,
    frameshift_ms: float = 10.0,
    datalen=None,
    f0min: float = 50.0,
    f0max: float = 500.0,
    windowsize_ms: float = 25.0,
    shr_threshold: float = 0.4,
    ceiling: float = 1250.0,
) -> pd.DataFrame:
    # normalize audio 
    y, fs = sf.read(str(wavfile), always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]
    y = y - np.mean(y)
    # mean value = 1
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    # set interpolation setting to choose FFT length larger than window size
    interpolation_depth = 0.5
    # determine how many rows the matrix will have
    N = int(np.floor(ceiling / f0min))
    N -= N % 2
    N *= 4
    # s --> ms for samples
    segmentlen = round(windowsize_ms * fs / 1000)
    inc        = round(frameshift_ms  * fs / 1000)
    total_len  = len(y)
    nf         = (total_len - segmentlen + inc) // inc
    # find center of each frame in ms
    f0_time_ms = (np.arange(nf) * frameshift_ms) + windowsize_ms / 2.0
    # FFT length
    fftlen = 1
    while fftlen < segmentlen * (1 + interpolation_depth):
        fftlen *= 2
    # build FFT-frequency axis in Hz
    # start at first positive bin
    frequency = fs * np.arange(1, fftlen // 2 + 1) / fftlen
    at_or_above_ceiling = np.where(frequency >= ceiling)[0]
    limit_idx = int(at_or_above_ceiling[0])
    # find limit freqyuency for FFT bins and keep frequencies up to that ceiling
    frequency = frequency[: limit_idx + 1]
    logf      = np.log2(frequency)
    # determine log frequency bin spacing
    # becomes the spacing used for the uniform log-frequency interpolation grid
    min_bin     = logf[-1] - logf[-2]
    shift_units = round(np.log2(N) / min_bin)
    # build log-frequency grid
    interp_logf = np.arange(logf[0], logf[-1] + min_bin / 2, min_bin)
    interp_len  = len(interp_logf)
    # define/transform candidate f0 search
    maxlogf = np.log2(f0max / 2)
    minlogf = np.log2(f0min / 2)
    upper_hits = np.where(interp_logf >= maxlogf)[0]
    lower_hits = np.where(interp_logf >= minlogf)[0]
    upperbound = int(upper_hits[0]) if len(upper_hits) else interp_len - 1
    lowerbound = int(lower_hits[0]) if len(lower_hits) else 0
    # convery back to hz
    newfre      = 2.0 ** interp_logf
    hamming_win = np.hamming(segmentlen)
    # convery frame centers from ms to samples 
    curpos      = np.round(f0_time_ms / 1000 * fs).astype(int)
    # allocate arrays 
    f0_raw  = np.zeros(nf)
    shr_raw = np.zeros(nf)
    cur_f0  = 0.0
    cur_shr = 0.0
    # main frame loop
    for n in range(nf):
        s = curpos[n] - segmentlen // 2
        s = max(0, min(s, total_len - segmentlen))
        segment = y[s : s + segmentlen] * hamming_win

        log_spec = _get_log_spectrum(segment, fftlen, limit_idx, logf, interp_logf)

        peak_index, cur_shr_frame = _compute_shr(
            log_spec, min_bin, N, shift_units, interp_len,
            lowerbound, upperbound, shr_threshold,
        )

        if peak_index != -1:
            cur_f0 = newfre[peak_index] * 2.0
            if cur_f0 > f0max:
                cur_f0 /= 2.0
            cur_shr = cur_shr_frame

        f0_raw[n]  = cur_f0
        shr_raw[n] = cur_shr

    if datalen is None:
        datalen = nf

    f0_out  = np.full(datalen, np.nan)
    shr_out = np.full(datalen, np.nan)

    t_rounded      = np.round(f0_time_ms).astype(int)
    frame_precision = 1

    for k_ms in range(0, int(t_rounded[-1]) + 1, int(frameshift_ms)):
        diffs = np.abs(t_rounded - k_ms)
        inx   = int(np.argmin(diffs))
        if diffs[inx] > frame_precision * frameshift_ms:
            continue
        idx = round(k_ms / frameshift_ms)
        if 0 <= idx < datalen:
            f0_out[idx]  = f0_raw[inx]
            shr_out[idx] = shr_raw[inx]

    return pd.DataFrame({"f0": f0_out, "shr": shr_out})
