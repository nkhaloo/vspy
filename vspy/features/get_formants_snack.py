"""
1. Load audio 
2. Preprocess it: Downsample and highpass filter
3. Cut into overlapping frames
4. Run LPC on each frame 
5. Convert LPC roots into poles and bandwidths 
6. Generate multiple formant and bandwidth candidates per frame 
7. Compute costs per candidate 
8. Find best path through frames that minimizes costs 
9. Return values in a DataFrame

"""
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from vspy.io import read_wav

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MAXCAN      = 300   # Maximum number of candidate pole-to-formant mappings per frame
MAXFORMANTS = 7     # Maximum number of formant slots (F1–F7)
MAXORDER    = 30    # Maximum LPC order

MISSING  = 1.0      # Penalty (in equivalent delta-Hz) when a formant has no pole assigned 
                    # basically a increases penalty for deviation from the nominal freq
                     
NOBAND   = 1000.0   # Penalty (in equivalent Hz bandwidth) for a missing formant;
                    # roughly one "formant slot" per 1000 Hz of the spectrum 
                    # basically gicing you a penalty as if you have a 1000 hz bandwidth

DF_FACT  = 20.0     # Weight on frame-to-frame frequency jump cost;
                    # higher values enforce smoother formant tracks

DFN_FACT = 0.3      # Weight on deviation from nominal formant frequencies (fnom);
                    # discourages tracks that stray too far from expected locations

BAND_FACT = 0.002   # Cost per Hz of pole bandwidth; prefers narrower, more
                    # resonance-like poles

F_BIAS   = 0.000    # Bias toward low-frequency poles 

F_MERGE  = 2000.0   # Penalty for mapping F1 and F2 to the same pole frequency

# Nominal formant frequencies (Hz) — used as soft targets when a candidate
# formant is missing; one entry per formant slot (F1–F7)
fnom  = np.array([500,  1500, 2500, 3500, 4500, 5500, 6500], dtype=np.float64)

# Allowed frequency range for each formant slot (Hz)
fmins = np.array([ 50,   400, 1000, 2000, 2000, 3000, 3000], dtype=np.float64)
fmaxs = np.array([1500, 3500, 4500, 5000, 6000, 6000, 8000], dtype=np.float64)

domerge = True  # flag controlling whether f1/f2 merge penalty is applied

# main function
def get_formants(signal: np.ndarray, samprate: float, frame_int: float,
                 lpc_ord: int, nform: int = 4,
                 ds_freq: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    # checks to see if you have enough LPC coef's to surrport # of formants 
    if nform > (lpc_ord - 4) // 2:
        raise ValueError(
            f"nform ({nform}) must be <= (lpc_ord - 4) // 2 = {(lpc_ord - 4) // 2}"
        )
    if nform > MAXFORMANTS:
        raise ValueError(f"nform must be <= {MAXFORMANTS}")
    sig, effective_rate = preprocess(signal, samprate, ds_freq)
    poles = lpc_poles(sig, effective_rate, frame_int, lpc_ord)
    return dpform(poles, nform, effective_rate)


# Approximate a decimal ratio (ds_freq/samprate) as numerator/denominator with
# denominator <= qlim, so resample_poly can up/downsample using small integers.
# e.g. 10000/44100 = 0.2268 -> returns (2, 9), i.e. upsample by 2, downsample by 9
def _ratprx(a: float, qlim: int = 10) -> tuple[int, int]:
    aa = abs(a)
    ai = int(aa)        # integer part of the ratio (0 for downsampling)
    af = aa - ai        # fractional part to approximate, e.g. 0.2268

    em, pp, qq = 1.0, 0, 0   # best error / numerator / denominator found so far

    # try every denominator q in [1, qlim] and keep the one whose nearest
    # numerator (ip = round(af * q)) gives the smallest error |af - ip/q|
    for q in range(1, qlim + 1):
        ps = q * af
        ip = int(ps + 0.5)          # round to nearest integer numerator
        e  = abs((ps - ip) / q)     # = |af - ip/q|, the approximation error
        if e < em:
            em, pp, qq = e, ip, q

    k = int(ai * qq + pp)   # fold the integer part back into the numerator
    return (-k if a < 0 else k), qq


# downsamples and highpass filters audio 
def preprocess(signal: np.ndarray, samprate: float,
               ds_freq: float = 10000.0) -> tuple[np.ndarray, int]:
    # Keep as float64 for downsampling 
    sig = np.asarray(signal, dtype=np.float64)
    if np.abs(sig).max() <= 1.0:      # normalize float [-1,1] to be in the int16 range
        sig = sig * 32767.0

    # only resample if the target rate (ratio_t) is lower than the input rate
    # ratio_t <= 0.99
    if ds_freq < samprate:
        # ratprx() finds finds the small interger ratio closest to ds_freq/samprate
        insert, decimate = _ratprx(ds_freq / samprate)
        ratio_t = insert / decimate
        # only resample if the rate change is meaningful (>1%); skip if ratio_t is ~1
        if ratio_t <= 0.99:
            sig      = resample_poly(sig, insert, decimate)
            samprate = samprate * ratio_t

    sig = sig.clip(-32768, 32767).astype(np.int16)

    # highpass()
    # filter legnth = 101 taps
    LCSIZ = 101
    # only compute hald of the coefficients
    ncoef = 1 + LCSIZ // 2   
    # hanning-like window to match Snack src                     
    fn    = np.pi * 2.0 / (LCSIZ - 1)   
    # scaling factor to fit in int16 range           
    scale = 32767.0 / (0.5 * LCSIZ)
    # build hanning-like low-pass filter 
    lcf   = (scale * (0.5 + 0.4 * np.cos(fn * np.arange(ncoef)))).astype(np.int16)

    # Spectral inversion: turns lowpass filter into a high pass by negating every coefficent and adding 2·Σlcf[1:] to the center tap
    # basically highpass = pass everything minus the lowpass
    co              = np.zeros(LCSIZ, dtype=np.int64)
    co[:ncoef - 1]  = -lcf[ncoef - 1:0:-1]          # -lcf[50], …, -lcf[1]
    co[ncoef:]      = -lcf[1:]                        # -lcf[1],  …, -lcf[50]
    co[ncoef - 1]   = 2 * int(np.sum(lcf[1:].astype(np.int64)))
    conv = np.convolve(sig.astype(np.int64), co, mode='same')
    out  = ((conv + 16384) >> 15).astype(np.int16)

    return out, int(samprate)


# takes freq (pole frequencies) and nform (how many formants you want), allocates a pc array, then runs candy to fill it
# basically chooses formant candidates from a series of poles
def get_fcand(freq: np.ndarray, nform: int) -> np.ndarray:
    # sorted pole frequencies for this frame
    freq  = np.asarray(freq, dtype=np.float64)
    # how many poles in this frame
    npole = len(freq)
    # storage for poles 
    pc    = np.full((MAXCAN, nform), -1, dtype=np.int16)
    # tracks index of highest used row? 
    ncan  = 0

# checks if pole pnumb falls within the allowed frequency range
    def canbe(pnumb, fnumb):
        return fmins[fnumb] <= freq[pnumb] <= fmaxs[fnumb]

# recursively fills rows of pc by assigning poles to formant slots as candidates
# cand = which row of pc we're in 
# pnumb = which pole we're considering 
# fnumb = which formant we're trying to fill 
    def candy(cand, pnumb, fnumb):
        nonlocal ncan
        # initialze with non values
        if fnumb < nform:
            pc[cand, fnumb] = -1

        if pnumb < npole and fnumb < nform:
            if canbe(pnumb, fnumb):
                pc[cand, fnumb] = pnumb

                # if a certain pole can possibly be f1 or f2, 
                # branch a new candiate row that passes the same pnumb
                # used later in F_merge penalty
                if domerge and fnumb == 0 and canbe(pnumb, fnumb + 1):
                    ncan += 1
                    pc[ncan, 0] = pc[cand, 0]
                    candy(ncan, pnumb, fnumb + 1)
                candy(cand, pnumb + 1, fnumb + 1)

                # if pnumb and pnumb + 1 (pole plus following pole) are both good candiates for fnumb
                # try both as potential candiates
                if (pnumb + 1) < npole and canbe(pnumb + 1, fnumb):
                    ncan += 1
                    pc[ncan, :fnumb] = pc[cand, :fnumb]
                    candy(ncan, pnumb + 1, fnumb)
            # if a certain pole can't be the formant we're on
            # move to next pole and keep trying to fill the same slot
            else:
                candy(cand, pnumb + 1, fnumb)
        # if we've run out of poles but there are stil formant slots
        # marks higher formants as missing instead of breaking
        if pnumb >= npole and fnumb < nform - 1 and pc[cand, fnumb] < 0:
            if fnumb:
                j = fnumb - 1
                while j > 0 and pc[cand, j] < 0:
                    j -= 1
                i = int(pc[cand, j]) if pc[cand, j] >= 0 else 0
            else:
                i = 0
            candy(cand, i, fnumb + 1)

    candy(0, 0, 0)
    # returns a 2D array where each row is a pole to formant candiate
    return pc[:ncan + 1]


# converts LPC coefs into pole frequencies and bandwidths in Hz 
# takes LPC coefs as input 
def formant(_lpc_ord: int, s_freq: float, lpca: np.ndarray,
            _init: bool) -> tuple[np.ndarray, np.ndarray]:
    # highest possible frequency
    nyquist = s_freq / 2.0
    # constant that converts radians to Hz
    pi2t    = 2.0 * np.pi / s_freq
    # finds all roots in an LPC. each root = 1 pole 
    roots = np.roots(lpca)

    freq_list = []
    band_list = []
    for r in roots:
        # keep conjugate from root with imag >= 0
        if r.imag < 0:         
            continue
        # skip a root sitting exactly at the origin
        if r.real == 0.0 and r.imag == 0.0:
            continue
        # find formant and convert from radians 
        theta = np.arctan2(r.imag, r.real)
        f = abs(theta) / pi2t
        # find bandwidth
        b = abs(0.5 * s_freq * np.log(r.real**2 + r.imag**2) / np.pi)
        # collect freq/band from each retained pole
        freq_list.append(f)
        band_list.append(b)
    # convert to arrays
    freq = np.array(freq_list)
    band = np.array(band_list)

    # split poles into junk poles (frequency ≈ 0 or ≈ Nyquist) and real poles 
    # sort real poles from low to high frequency 
    # dump unsorted poles at the end
    is_complex = (freq > 1.0) & (freq < nyquist)
    complex_idx = np.where(is_complex)[0]
    real_idx    = np.where(~is_complex)[0]
    order       = np.concatenate([complex_idx[np.argsort(freq[complex_idx])], real_idx])
    freq, band  = freq[order], band[order]

    return freq, band


# computes LPC coefficients using weighted covariance (essentially builds a covariation matrix)
# LPC_ord = LPC order (p) (# of LPC coef's being solved)
# wind = window sizde in samples 
# data = frame's audio samples 
# premp = pre-emphasis coefficients
def lpcbsa(lpc_ord: int, wind: int, data: np.ndarray, preemp: float):
    owind = wind

    # Hamming window
    # represents weighting: values in the center of the signal matter most
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(owind) / owind)

    # add extra samples to make buffer (stores frames audio samples) larger than window
    total = owind + lpc_ord + 1

    # copy data into buffer, zero-pad if short, add dither noise
    sig = np.zeros(total, dtype=np.float64)
    n = min(len(data), total)
    sig[:n] = data[:n].astype(np.float64)
    sig += 0.016 * np.random.uniform(size=total) - 0.008

    # pre-emphasis
    sig[:-1] = sig[1:] - preemp * sig[:-1]
    sig = sig[:owind + lpc_ord]  

    # RMS energy normalization over the entire buffer (lpc_ord + owind)
    energy = float(np.sqrt(np.dot(sig[lpc_ord:lpc_ord + owind],
                                  sig[lpc_ord:lpc_ord + owind]) / owind))
    if energy <= 0.0:
        return np.zeros(lpc_ord + 1), 0.0
    sig /= energy

    # weighted covariance matrix 
    # covariance LPC 
    p   = lpc_ord
    # correlation  between delayed copies of the signal
    phi = np.zeros((p, p))
    # correlation between the current signal and each delayed copy
    shi = np.zeros(p)
    # current signal segment
    y   = sig[p:p + owind]

    for i in range(p):
        # delayed version of y
        si = sig[p - i - 1: p - i - 1 + owind]
        for j in range(i + 1):
            # delayed y 
            sj = sig[p - j - 1: p - j - 1 + owind]
            v = float(np.dot(si * sj, w))
            phi[i, j] = phi[j, i] = v
        # how correlated is the unshifted signal with the signal shifted by i? 
        shi[i] = float(np.dot(y * si, w))

    # banded ridge regularization 
    # adds a penalty that keeps the coefficients from changing a lot from one to the next 
    pss = float(np.dot(y ** 2, w))
    pre = 0.09 * pss
    # these particular weights (0.375, -0.25, 0.0625) encode a second-difference operator
    # matches Snack src smoothing
    for i in range(p):
        phi[i, i] += 0.375 * pre
        if i > 0:
            phi[i, i - 1] -= 0.25 * pre
            phi[i - 1, i] -= 0.25 * pre
        if i > 1:
            phi[i, i - 2] += 0.0625 * pre
            phi[i - 2, i] += 0.0625 * pre

    try:
        # LPC coefficients 
        a = np.linalg.solve(phi, shi)
    except np.linalg.LinAlgError:
        return np.zeros(lpc_ord + 1), energy

    # A(z) = 1 - a[0]*z^-1 - ... - a[p-1]*z^-p  →  decreasing-power coefficients
    return np.concatenate([[1.0], -a]), energy


# takes raw audio and returns per-frame resonant poles and bandwidths
def lpc_poles(signal: np.ndarray, samprate: float, frame_int: float,
              lpc_ord: int) -> list[dict]:
    import math
    # prevents invalid LPC orders
    if lpc_ord > MAXORDER or lpc_ord < 2:
        raise ValueError(f"lpc_ord must be between 2 and {MAXORDER}")
    # sets window duration and preemphasis coefficient
    wdur   = 0.025
    preemp = math.exp(-62.831853 * 90.0 / samprate)   # exp(-1800*pi*T)
    # convery duration to samples
    size = round(wdur * samprate)
    step = round(frame_int * samprate)
    # number of frames
    nfrm = 1 + int((len(signal) / samprate - size / samprate) / (step / samprate))
    if nfrm < 1:
        raise ValueError("Signal too short for given window/frame parameters")

    data  = signal.astype(np.int16)
    poles = []
    init  = True
    # frame loop 
    for j in range(nfrm):
        start = j * step
        frame = data[start:start + size]

        lpca, energy = lpcbsa(lpc_ord, size, frame, preemp)
        # convert LPC coefficients to poles
        if energy > 1.0:
            freq, band = formant(lpc_ord, samprate, lpca, init)
            init = False
        else:
            freq = np.empty(0)
            band = np.empty(0)
            init = True

        poles.append({'rms': energy, 'freq': freq, 'band': band, 'npoles': len(freq)})

    return poles


# returns maximum RMS energy value across all frames. used later for normalization
def get_stat_max(rms: np.ndarray) -> float:
    return float(np.max(rms))


# Dynamic programming: find the best path (left to right) for formants across all frames of an audio file
def dpform(poles: list[dict], nform: int, samprate: float):
    nframes = len(poles)

    # scale cost weights to sample rate
    dffact     = DF_FACT   * 0.01 * samprate
    bfact      = BAND_FACT / (0.01 * samprate)
    ffact      = DFN_FACT  / (0.01 * samprate)
    FBIAS      = F_BIAS    / (0.01 * samprate)
    merge_cost = F_MERGE
    global domerge
    if merge_cost > 1000.0:
        domerge = False

    rmsmax = get_stat_max(np.array([p['rms'] for p in poles]))

    # output arrays
    fr = np.zeros((nform, nframes))
    ba = np.zeros((nform, nframes))

    # DP lattice: one dict per frame with keys ncand, cand, cumerr, prept, freq, band
    fl: list[dict] = [{'ncand': 0, 'cand': np.empty((0, nform), dtype=np.int16),
                        'cumerr': np.empty(0), 'prept': np.empty(0, dtype=np.int32),
                        'freq': np.empty(0), 'band': np.empty(0)}
                       for _ in range(nframes)]

    # ------------------------------------------------------------------ #
    # Forward pass: build candidates and cumulative costs                  #
    # ------------------------------------------------------------------ #
    for i, pole in enumerate(poles):
        freq = pole['freq']
        band = pole['band']

        # scale frequency-jump cost by relative RMS (louder = cheaper to jump)
        rmsdffact = (pole['rms'] / rmsmax) * dffact

        if len(freq) > 0:
            cands = get_fcand(freq, nform)   # (ncand, nform)
        else:
            cands = np.empty((0, nform), dtype=np.int16)

        ncand  = len(cands)
        prept  = np.full(ncand, -1, dtype=np.int32)
        cumerr = np.zeros(ncand)

        for j in range(ncand):
            # --- transition cost: best connection to previous frame ---
            if i > 0:
                prev = fl[i - 1]
                minerr = 2e30 if prev['ncand'] > 0 else 0.0
                mincan = -1
                for k in range(prev['ncand']):
                    pferr = 0.0
                    for fi in range(nform):
                        ic = int(cands[j, fi])
                        ip = int(prev['cand'][k, fi])
                        if ic >= 0 and ip >= 0:
                            ft = 2.0 * abs(freq[ic] - prev['freq'][ip]) / (freq[ic] + prev['freq'][ip])  # noqa: E501
                            pferr += ft * ft
                        else:
                            pferr += MISSING
                    conerr = rmsdffact * pferr + prev['cumerr'][k]
                    if conerr < minerr:
                        minerr = conerr
                        mincan = k
            else:
                minerr = 0.0
                mincan = -1

            prept[j] = mincan

            # --- local costs ---
            berr  = 0.0
            ferr  = 0.0
            fbias = 0.0
            merger = 0.0
            for k in range(nform):
                ic = int(cands[j, k])
                if ic >= 0:
                    if k == 0 and domerge:
                        ic1 = int(cands[j, 1]) if nform > 1 else -1
                        if ic1 >= 0 and freq[ic] == freq[ic1]:
                            merger = merge_cost
                    berr  += band[ic]
                    ferr  += abs(freq[ic] - fnom[k]) / fnom[k]
                    fbias += freq[ic]
                else:
                    berr  += NOBAND
                    ferr  += MISSING
                    fbias += fnom[k]

            cumerr[j] = FBIAS * fbias + bfact * berr + merger + ffact * ferr + minerr

        fl[i] = {'ncand': ncand, 'cand': cands, 'cumerr': cumerr,
                 'prept': prept, 'freq': freq, 'band': band}

    # ------------------------------------------------------------------ #
    # Backtrack: find min-cost path from last frame to first               #
    # ------------------------------------------------------------------ #
    mincan = -1
    for i in range(nframes - 1, -1, -1):
        frame = fl[i]

        if mincan < 0 and frame['ncand'] > 0:
            mincan = int(np.argmin(frame['cumerr']))

        if mincan >= 0:
            for j in range(nform):
                k = int(frame['cand'][mincan, j])
                if k >= 0:
                    fr[j, i] = frame['freq'][k]
                    ba[j, i] = frame['band'][k]
                else:
                    if i < nframes - 1:
                        fr[j, i] = fr[j, i + 1]  # replicate backwards
                        ba[j, i] = ba[j, i + 1]
                    else:
                        fr[j, i] = fnom[j]
                        ba[j, i] = NOBAND
            mincan = int(frame['prept'][mincan])
        else:
            for j in range(nform):
                fr[j, i] = fnom[j]
                ba[j, i] = NOBAND

    return fr, ba

# entry point wrapper 
def get_formants_snack(wavfile, frameshift_ms=1, datalen=None,
                       lpc_ord=12, nform=4, ds_freq=10000.0):
    # read wav file 
    y, fs = read_wav(wavfile)
    # convert from seconds to ms
    frame_int = frameshift_ms / 1000.0
    # main tracker is called
    fr, ba = get_formants(y, fs, frame_int, lpc_ord, nform, ds_freq)
    nframes = fr.shape[1]

    if datalen is None:
        datalen = nframes

    n = min(datalen, nframes)
    cols = {}
    for k in range(nform):
        Fk = np.full(datalen, np.nan)
        Bk = np.full(datalen, np.nan)
        Fk[:n] = fr[k, :n]
        Bk[:n] = ba[k, :n]
        cols[f"F{k + 1}_snack"] = Fk
        cols[f"B{k + 1}_snack"] = Bk

    t_ms = np.arange(datalen, dtype=float) * frameshift_ms
    return pd.DataFrame({"t_ms": t_ms, **cols})


