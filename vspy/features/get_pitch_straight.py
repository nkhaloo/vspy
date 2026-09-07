"""Python transcription of VoiceSauce's TANDEM-STRAIGHT F0 path.

Algorithm source: Hideki Kawahara, TandemSTRAIGHTmonolithicPackage014
(TANDEMf0GB, TandemSTRAIGHTGeneralBody, tandem2STRAIGHT,
exF0candidatesTSTRAIGHTGB, baseF0withIF, refineF0AssumptionGB,
autoF0Tracking). Wrapper semantics follow UCLA VoiceSauce func_StraightPitch.
The upstream package's READMEfirst.txt supplies its license terms; this is
not the Apache-licensed legacy STRAIGHT algorithm.

F0 is deliberately NOT masked by the separate voicing decision: VoiceSauce
returns the tracked, nine-point-smoothed F0 even in unvoiced regions.
Numerical equivalence is evaluated against the exported VoiceSauce CSV;
bit identity with MATLAB floating-point intermediates is not asserted.
"""
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import cheby1, filtfilt, lfilter

from vspy.io import read_wav


def _round(x):
    x = np.asarray(x)
    return np.copysign(np.floor(np.abs(x) + 0.5), x).astype(np.int64)


def _interp(x, y, xi):
    return interp1d(x, y, axis=-1, bounds_error=False,
                    fill_value='extrapolate', assume_sorted=False)(xi)


def _frame_times(duration):
    # MATLAB colon constructs the two halves from opposite endpoints. These
    # last-bit differences matter when a window lands on a half sample.
    count = int(np.floor(duration / 0.001 + 0.5))
    end = count * 0.001
    tolerance = 2 * np.finfo(float).eps * abs(duration)
    if abs(end - duration) <= tolerance:
        end = duration
    elif end > duration:
        count -= 1
        end = count * 0.001
    times = np.empty(count + 1)
    half = count // 2
    steps = np.arange(half + 1) * 0.001
    times[:half + 1] = steps
    times[count-half:] = end - steps[::-1]
    if count % 2 == 0:
        times[half] = end / 2
    return times


def _decimate(x, q):
    if q == 1:
        return x.copy()
    b, a = cheby1(8, 0.05, 0.8 / q)
    # MATLAB filtfilt uses 3*(max(len(a),len(b))-1) reflected samples.
    # MATLAB decimate retains samples backwards from the final sample, then
    # reverses them. The phase therefore depends on the segment length.
    return filtfilt(b, a, x, padlen=3 * (max(len(a), len(b)) - 1))[(len(x)-1) % q::q]


def _spectrum(x, fs, times, f0, nfft):
    """TandemSTRAIGHTGeneralBody + its power-domain rectangular smoother."""
    base = np.arange(-int(_round(2 * fs / f0)), int(_round(2 * fs / f0)) + 1)
    power = np.zeros((len(times), nfft))
    for sign in (-1, 1):
        center = (times + sign / f0 / 4) * fs
        indices = np.clip(_round(center[:, None] + 1 + base) - 1, 0, len(x) - 1)
        t = base[None, :] / fs / 2 + (center - _round(center))[:, None] / fs
        window = 0.5 * np.cos(np.pi * t * f0) + 0.42 + 0.08 * np.cos(2 * np.pi * t * f0)
        window /= np.sqrt(np.sum(window ** 2, axis=1))[:, None]
        segment = x[indices] * window
        segment -= (np.mean(segment, axis=1) / np.mean(window, axis=1))[:, None] * window
        power += np.abs(np.fft.fft(segment, n=nfft, axis=1)) ** 2
    frequency = np.arange(nfft) / nfft * fs
    low = frequency < 1.2 * f0
    replica = _interp(f0 - frequency[low], power[:, low], frequency[low])
    count = np.count_nonzero(frequency < f0)
    power[:, :count] += replica[:, :count]
    power[:, nfft // 2 + 1:] = power[:, 1:nfft // 2][:, ::-1]
    integral = np.cumsum(np.tile(power, (1, 2)) * (fs / nfft), axis=1)
    double_axis = np.arange(2 * nfft) / nfft * fs - fs + fs / nfft / 2
    centers = np.arange(nfft // 2 + 1) / nfft * fs

    def interp_h(xi):
        # Preserve interp1H's regular-grid arithmetic, rather than interp1.
        pos = (np.clip(xi, double_axis[0], double_axis[-1]) - double_axis[0]) / (double_axis[1] - double_axis[0])
        index = np.floor(pos).astype(int)
        return integral[:, index] + (integral[:, np.minimum(index + 1, 2*nfft-1)] - integral[:, index]) * (pos - index)

    straight = np.maximum(100 * np.finfo(float).eps,
                          (interp_h(centers + f0 / 2) - interp_h(centers - f0 / 2)) / f0)
    return straight, power[:, :nfft // 2 + 1]


def _periodicity(x, fs, times, floor, ceiling):
    nfft = 2 ** int(np.ceil(np.log2(4 * fs / floor + 1)))
    bands = floor * 2 ** (np.arange(int(np.ceil(np.log2(ceiling / floor) * 3)) + 1) / 3)
    lag = np.arange(nfft // 2 + 1) / fs
    lag[0] = 0.1 / fs
    cumulative = np.zeros((len(times), len(lag)))
    powers = []
    for band in bands:
        straight, tandem = _spectrum(x, fs, times, band, nfft)
        period = band / fs * nfft
        length = min(nfft // 2 + 1, int(np.floor(period * 4)) + 1)
        filt = 0.5 + 0.5 * np.cos(np.pi * np.arange(length) / period / 4)
        filt /= np.sum(filt)
        weighted = np.zeros((len(times), nfft))
        weighted[:, :length] = (tandem[:, :length] / straight[:, :length] - 1) * filt
        weighted[:, -(length - 1):] = weighted[:, 1:length][:, ::-1]
        correlation = np.fft.fft(weighted, axis=1).real
        loglag = np.log2(lag * band) * 2.5
        weight = (0.5 + 0.5 * np.cos(np.pi * loglag)) * (np.abs(loglag) < 1)
        cumulative += correlation[:, :len(lag)] * weight
        powers.append(np.sum(straight[:, :length] * filt, axis=1))
    powers = np.asarray(powers).T
    power = _interp(1 / np.r_[bands[0] * 0.9, bands, bands[-1] * 1.1],
                    np.column_stack((powers[:, 0], powers, powers[:, -1])), lag)
    return cumulative, power


@lru_cache(maxsize=16)
def _calibration(fs, floor, ceiling):
    x = np.zeros(int(np.floor(0.2 * fs)) + 1)
    x[::int(_round(fs / np.sqrt(floor * ceiling)))] = 1
    x -= np.mean(x)
    periodicity, _ = _periodicity(x, fs, np.array([0.1]), floor, ceiling)
    return np.max(periodicity)


def _base_if(x, fs, time, f0):
    half = int(_round(1.5 * fs / f0))
    base = np.arange(-half, half + 1)
    center = int(_round(time * fs))
    segment = x[np.clip(base + center, 0, len(x) - 1)]
    t = base / fs + (time * fs - center) / fs
    envelope = 0.5 * np.cos(np.pi * t / 1.5 * f0) + 0.42 + 0.08 * np.cos(2 * np.pi * t / 1.5 * f0)
    harmonics = np.arange(1, 4)
    # MATLAB ' is conjugate transpose, not just transpose.
    quadrature = envelope * np.exp(-1j * 2 * np.pi * f0 * t * harmonics[:, None])
    s = np.sum(segment * quadrature, axis=1)
    s1 = np.sum(segment * quadrature * t, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        freq = f0 * harmonics + (s.real * s1.imag - s.imag * s1.real) / np.abs(s)**2 * fs / 2 / np.pi
        weights = np.sqrt(np.abs(s))
        return np.sum(weights * freq / harmonics) / np.sum(weights)


def _refine(x, fs, time, f0):
    high, low = f0 / 0.995, f0 * 0.995
    values = [_base_if(x, fs, time, high), _base_if(x, fs, time, low)]
    a = np.linalg.inv(np.array([[high, 1], [low, 1]])) @ values
    with np.errstate(divide='ignore', invalid='ignore'):
        return a[1] / (1 - a[0])


def _track(f0, candidates, scores, powers, times):
    """F0-producing portion of autoF0Tracking (VUV does not change F0)."""
    raw = candidates[:, 0]
    tuned = f0.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        bad = np.abs(np.log2(tuned / raw)) > 0.5
    tuned[bad] = raw[bad]
    top = scores[:, 0]
    with np.errstate(divide='ignore'):
        power = 10 * np.log10(powers[:, 0])
    reliable = top > 0.65
    if not reliable.any():
        return f0
    level = np.median(top[reliable])
    sd = np.std(top[reliable], ddof=1) if reliable.sum() > 1 else 0.0
    voiced_power = np.median(power[reliable])
    edges = np.diff(np.r_[False, top > level - 2 * sd, False].astype(int))
    on, off = np.flatnonzero(edges > 0), np.flatnonzero(edges < 0)
    keep = off - on > 1
    on, off = on[keep], off[keep]
    keep = times[off - 1] - times[on] > 0.019
    on, off = on[keep], np.minimum(len(times) - 1, off[keep])
    keep = np.array([np.mean(power[a:b+1]) > voiced_power - 30 for a, b in zip(on, off)], dtype=bool)
    on, off = on[keep], off[keep]
    if not len(on):
        return f0
    skip = (2 ** (1/3) - 1) * np.sqrt(np.mean(np.diff(times)) / 0.005)
    skeleton = np.zeros(len(times))
    skeleton_score = skeleton.copy()
    backward = 0
    forward = int(np.ceil((on[1] + off[0]) / 2)) if len(on) > 1 else len(times) - 1
    for i in range(len(on)):
        center = int(_round((on[i] + off[i]) / 2))
        start = raw[center]
        for positions in (range(center, backward - 1, -1), range(center, forward + 1)):
            previous = start
            for j in positions:
                closest = np.argmin(np.abs(candidates[j] - previous))
                distance = abs(candidates[j, closest] - previous)
                if distance < skip * previous and scores[j, closest] > 0.2:
                    previous = candidates[j, closest]
                skeleton[j] = previous
                skeleton_score[j] = scores[j, closest]
        backward = forward
        forward = int(np.ceil((on[i+2] + off[i+1]) / 2)) if len(on) > i + 2 else len(times) - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        bad = np.abs(np.log2(tuned / skeleton)) > 0.5
    tuned[bad] = skeleton[bad]
    skeleton[skeleton_score > 0.8] = tuned[skeleton_score > 0.8]
    return skeleton


def straight_f0(x, fs, min_f0=40, max_f0=500):
    """VoiceSauce calculateF0 at 1 ms, before length padding/downsampling."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('STRAIGHT requires finite mono audio')
    if not 0 < min_f0 < max_f0 < fs / 2:
        raise ValueError('Require 0 < min_f0 < max_f0 < Nyquist')
    if not len(x):
        return np.empty(0)
    q = max(1, int(_round(fs / 6000)))
    if len(x) <= 24 or not np.any(x):
        return np.full(int(np.floor((len(x)-1) / fs * 1000)) + 1, np.nan)
    xd, fsd = _decimate(x, q), fs / q
    times = _frame_times((len(xd) - 1) / fsd)
    # Bound temporary spectrum memory without changing frame calculations.
    blocks = [_periodicity(xd, fsd, times[i:i+256], min_f0, max_f0) for i in range(0, len(times), 256)]
    periodicity = np.concatenate([p[0] for p in blocks])
    power = np.concatenate([p[1] for p in blocks])
    calibration = _calibration(fsd, min_f0, max_f0)
    candidates = np.zeros((len(times), 5))
    scores, powers = candidates.copy(), candidates.copy()
    refined = np.empty(len(times))
    for i, (time, scan) in enumerate(zip(times, periodicity)):
        peak = np.argmax(scan)
        left, right = scan[max(0, peak-1)] - scan[peak], scan[min(len(scan)-1, peak+1)] - scan[peak]
        lag = peak if left + right == 0 else peak - (right-left) / (left+right) / 2
        raw = fsd / (lag - 0.07)
        value = np.clip(raw, min_f0, fsd / 2)
        for _ in range(2):
            value = _refine(xd, fsd, time, value)
            # MATLAB min/max omit NaN for scalar bounds.
            value = np.fmax(min_f0, np.fmin(fsd / 2, value))
        refined[i] = value
        before, after = np.diff(np.r_[0, scan]), np.diff(np.r_[scan, 0])
        peaks = np.flatnonzero((before * after < 0) & (before > 0))
        ym = scan[np.maximum(0, peaks-1)] - scan[peaks]
        yp = scan[np.minimum(len(scan)-1, peaks+1)] - scan[peaks]
        lags = peaks - (yp-ym) / (ym+yp) / 2
        frequencies = fsd / (lags - 0.07)
        order = np.argsort(-scan[peaks], kind='stable')
        count = min(5, np.count_nonzero(frequencies > 0))
        selected = order[:count]
        candidates[i, :count] = frequencies[selected]
        scores[i, :count] = scan[peaks[selected]] / calibration
        powers[i, :count] = power[i, peaks[selected]]
    f0 = _track(refined, candidates, scores, powers, times)
    if len(f0) > 14:
        f0 = lfilter(np.ones(9) / 9, 1, np.r_[np.repeat(f0[0], 9), f0, np.repeat(f0[-1], 4)])[13:]
    f0[f0 == 0] = np.nan
    return f0


def get_pitch_straight(wavfile, frameshift_ms=1, datalen=None,
                       min_f0=40, max_f0=500, intervals=None, tbuffer=25,
                       ignore_labels=('', ' ', 'SIL')):
    """Registry extractor; intervals are (start_ms, end_ms, label).

    With intervals, mirror func_StraightPitch's one-based segment extraction
    and placement. Labels in VoiceSauce's default ignore list are skipped.
    Without intervals, analyze the entire signal (no arbitrary chunk seams).
    """
    if frameshift_ms < 1 or int(frameshift_ms) != frameshift_ms:
        raise ValueError('STRAIGHT frameshift_ms must be a positive integer')
    y, fs = read_wav(wavfile)
    if datalen is None:
        datalen = int(len(y) / fs * 1000 / frameshift_ms)
    if intervals is None:
        f0 = straight_f0(y, fs, min_f0, max_f0)
    else:
        length = max(0, int(len(y) / fs * 1000) - 1)
        f0 = np.full(length, np.nan)
        for start, stop, label in intervals:
            if label in ignore_labels:
                continue
            start, stop = start - tbuffer, stop + tbuffer
            first = max(1, int(np.floor(start / 1000 * fs)))
            last = min(len(y), int(np.ceil(stop / 1000 * fs)))
            if last < first:
                continue
            segment = straight_f0(y[first-1:last], fs, min_f0, max_f0)
            offset = max(1, int(np.floor(start))) - 1
            count = min(len(segment), max(0, length - offset))
            f0[offset:offset+count] = segment[:count]
    f0 = f0[::int(frameshift_ms)]
    output = np.full(datalen, np.nan)
    output[:min(datalen, len(f0))] = f0[:datalen]
    return output
