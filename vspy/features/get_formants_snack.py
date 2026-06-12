"""
1. Load audio 
2. Preprocess it: Downsample and highpass filter
3. Cut into overlapping frames
4. Run LPC on each frame 
5. Convert LPC roots into resonances and bandwidths 
6. Generate multiple formant and bandwidth candidates per frame 
7. Compute costs per candidate 
8. Find best path through frames that minimizes costs 
9. Return values in a DataFrame

"""
import math
import numpy as np
import pandas as pd
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

# Root finder selection. Snack factors the LPC polynomial with a Lin-Bairstow
# solver (lbpoly); we ported it (below) to test whether its finite-tolerance /
# deflation-order / warm-start error signature was the source of the bandwidth
# mismatch. RESULT: it was NOT — on the eval set lbpoly reproduced np.roots to
# ~1e-4 and changed no metric. Lin-Bairstow converges to the same true roots
# np.roots finds, so the bandwidth gap lives upstream (in the LPC coefficients /
# pole selection), not in root finding. Default left on np.roots (faster); flip
# to True to re-run the A/B. The lbpoly port is kept for reference/faithfulness.
USE_LBPOLY = False

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
# denominator <= qlim, so we can up/downsample using small integers.
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


# Symmetric Hanning-windowed-sinc lowpass FIR design, matching Snack's lc_lin_fir().
# Returns the half-kernel coef[0..n-1] where coef[0] is the center tap.
def _lc_lin_fir(fc: float, nf: int = 127) -> np.ndarray:
    n = (nf + 1) // 2
    coef = np.zeros(n)
    coef[0] = 2.0 * fc
    i = np.arange(1, n)
    coef[1:] = np.sin(i * 2.0 * np.pi * fc) / (np.pi * i)
    fn = 2.0 * np.pi / (nf - 1)
    coef *= (0.5 + 0.5 * np.cos(fn * np.arange(n)))
    return coef


# Exact port of Snack's do_fir() (downsample.c). `ic` holds the half-kernel
# (ic[0] = center tap); the full symmetric kernel of length 2*ncoef-1 is built
# here. invert=1 turns the lowpass into a highpass by spectral inversion.
# CRITICAL: Snack rounds and >>15-shifts EACH tap product before summing
# (sum += (co[t]*mem[t] + 16384) >> 15), not the final dot product. Summing in
# full precision and shifting once (np.convolve) gives different int16 samples,
# which perturbs every frame's LPC. We reproduce the per-tap rounding exactly.
# The result is stored to int16 with wraparound (C stores into short), not clip.
def _do_fir(buf: np.ndarray, ic: np.ndarray, ncoef: int, invert: bool) -> np.ndarray:
    ic = np.asarray(ic, dtype=np.int64)
    k  = 2 * ncoef - 1
    co = np.empty(k, dtype=np.int64)
    if not invert:
        co[:ncoef - 1] = ic[ncoef - 1:0:-1]          # ic[n-1],...,ic[1]
        co[ncoef - 1]  = ic[0]                        # center (point of symmetry)
        co[ncoef:]     = ic[1:ncoef]                  # ic[1],...,ic[n-1]
    else:
        co[:ncoef - 1] = -ic[ncoef - 1:0:-1]
        co[ncoef - 1]  = 2 * int(ic[1:ncoef].sum())   # = integral - ic[0]
        co[ncoef:]     = -ic[1:ncoef]

    buf = np.asarray(buf, dtype=np.int64)
    n   = len(buf)
    # mem starts as [ncoef-1 zeros | buf[0:ncoef]] and slides one sample per
    # output; output o convolves co with ext[o:o+k]. The trailing ncoef zeros
    # are the zero-padded tail Snack appends after the main loop.
    ext = np.concatenate([np.zeros(ncoef - 1, dtype=np.int64), buf,
                          np.zeros(ncoef, dtype=np.int64)])
    out = np.zeros(n, dtype=np.int64)
    for t in range(k):                                # accumulate tap by tap (O(k) passes, O(n) mem)
        out += (co[t] * ext[t:t + n] + 16384) >> 15
    return out.astype(np.int16)                       # short cast == wraparound


# Snack's quick-and-dirty downsampler (Fdownsample/dwnsamp in jkFormant.c):
# zero-insert to upsample by `insert`, lowpass with a Hanning-windowed sinc
# at the new Nyquist, then decimate by `decimate`. Operates on int16 audio
# with the same fixed-point (>>15) arithmetic as the C source.
def _downsample_snack(sig: np.ndarray, samprate: float,
                       insert: int, decimate: int) -> tuple[np.ndarray, float]:
    ratio_t = insert / decimate
    freq2   = ratio_t * samprate

    # cutoff of the lowpass, normalized to the upsampled rate (= new Nyquist)
    beta = 0.5 / decimate
    b    = _lc_lin_fir(beta)

    # quantize to int16 coefficients, matching (int)(0.5 + 32767*b[i])
    maxi = 32767.0
    ic = np.trunc(maxi * b + 0.5).astype(np.int64)
    nz = np.nonzero(ic)[0]
    ncoef = int(nz[-1]) + 1 if len(nz) else 1
    ic = ic[:ncoef]

    in_samps = len(sig)
    imax = int(np.max(np.abs(sig))) if in_samps else 0
    if imax == 0:
        imax = 1
    # rescale input toward full int16 range before zero-insertion
    if insert > 1:
        k = (32767 * 32767) // imax
    else:
        k = (16384 * 32767) // imax
    scaled = (k * sig.astype(np.int64) + 16384) >> 15

    up = np.zeros(in_samps * insert, dtype=np.int64)
    up[::insert] = scaled

    filtered = _do_fir(up, ic, ncoef, invert=False)   # int16, per-tap rounded

    out_samps = (in_samps * insert) // decimate
    out = filtered[:out_samps * decimate:decimate]

    return out.astype(np.int16), freq2


# downsamples and highpass filters audio
def preprocess(signal: np.ndarray, samprate: float,
               ds_freq: float = 10000.0) -> tuple[np.ndarray, int]:
    # Keep as float64 for downsampling
    sig = np.asarray(signal, dtype=np.float64)
    if np.abs(sig).max() <= 1.0:      # normalize float [-1,1] to be in the int16 range
        sig = sig * 32767.0

    # NOTE: VoiceSauce's "process at 16kHz" resample+quantize step happens once,
    # globally, in api.py before any feature extractor runs (matching
    # vs_ParameterEstimation.m, which resamples/rewrites the wav before
    # dispatching to any feature). Do not repeat it here.
    sig = sig.clip(-32768, 32767).round().astype(np.int16)

    # only resample if the target rate (ratio_t) is lower than the input rate
    # ratio_t <= 0.99
    if ds_freq < samprate:
        # ratprx() finds finds the small interger ratio closest to ds_freq/samprate
        insert, decimate = _ratprx(ds_freq / samprate)
        ratio_t = insert / decimate
        # only resample if the rate change is meaningful (>1%); skip if ratio_t is ~1
        if ratio_t <= 0.99:
            sig, samprate = _downsample_snack(sig, samprate, insert, decimate)

    # highpass()
    # filter legnth = 101 taps
    LCSIZ = 101
    # only compute hald of the coefficients
    ncoef = 1 + LCSIZ // 2
    # hanning-like window to match Snack src
    fn    = np.pi * 2.0 / (LCSIZ - 1)
    # scaling factor to fit in int16 range
    scale = 32767.0 / (0.5 * LCSIZ)
    # build hanning-like low-pass half-kernel (lcf[0] = center)
    lcf   = (scale * (0.5 + 0.4 * np.cos(fn * np.arange(ncoef)))).astype(np.int16)

    # Spectral inversion (invert=1) turns the lowpass into a highpass; run it
    # through the exact same per-tap-rounded FIR Snack uses (do_fir).
    out = _do_fir(sig, lcf, ncoef, invert=True)

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


# ---------------------------------------------------------------------------
# Lin-Bairstow root finder (port of lbpoly()/qquad() from Snack's sigproc2.c).
# Used in place of np.roots so pole radii (=> bandwidths) match Snack's finite-
# tolerance, deflation-ordered, warm-started solver rather than improving on it.
# ---------------------------------------------------------------------------

_LB_MAX_ITS  = 100       # max Newton iterations per quadratic factor
_LB_MAX_TRYS = 100       # max random restarts before giving up on a factor
_LB_MAX_ERR  = 1.0e-6    # acceptable residual |b0| + |b1| on the quad factor
_LB_DBL_MAX  = 1.7976931348623157e308
_LB_LIM0     = 0.5 * math.sqrt(_LB_DBL_MAX)


def _qquad(a: float, b: float, c: float):
    """Solve a*x^2 + b*x + c = 0; returns (r1r, r1i, r2r, r2i, ok)."""
    if a == 0.0:
        if b == 0.0:
            return 0.0, 0.0, 0.0, 0.0, False
        return -c / b, 0.0, 0.0, 0.0, True
    numi = b * b - 4.0 * a * c
    if numi >= 0.0:
        # use the numerically stabler form (avoid cancellation in -b ± sqrt)
        if b < 0.0:
            y = -b + math.sqrt(numi)
            return y / (2.0 * a), 0.0, (2.0 * c) / y, 0.0, True
        y = -b - math.sqrt(numi)
        return (2.0 * c) / y, 0.0, y / (2.0 * a), 0.0, True
    den = 2.0 * a
    r1i = math.sqrt(-numi) / den
    r   = -b / den
    return r, r1i, r, -r1i, True


def _lbpoly(a: list, order: int, rootr: list, rooti: list) -> bool:
    """Find the `order` roots of the polynomial whose coefficients are `a` in
    INCREASING power (a[0] = constant term, a[order] = leading), via Lin-Bairstow
    deflation. `a` is consumed (deflated) in place; pass a copy. `rootr`/`rooti`
    carry starting guesses on entry (warm start) and receive the roots on exit."""
    b = [0.0] * (order + 1)
    c = [0.0] * (order + 1)
    ord_ = order
    while ord_ > 2:
        ordm1 = ord_ - 1
        ordm2 = ord_ - 2
        # kluge from the C source: zero out near-zero leftover roots to dodge underflow
        if abs(rootr[ordm1]) < 1.0e-10:
            rootr[ordm1] = 0.0
        if abs(rooti[ordm1]) < 1.0e-10:
            rooti[ordm1] = 0.0
        p = -2.0 * rootr[ordm1]                                  # quad factor x^2 + p x + q
        q = rootr[ordm1] * rootr[ordm1] + rooti[ordm1] * rooti[ordm1]
        found = False
        for _ntrys in range(_LB_MAX_TRYS):
            found = False
            for _itcnt in range(_LB_MAX_ITS):
                lim = _LB_LIM0 / (1.0 + abs(p) + abs(q))
                b[ord_]  = a[ord_]
                b[ordm1] = a[ordm1] - p * b[ord_]
                c[ord_]  = b[ord_]
                c[ordm1] = b[ordm1] - p * c[ord_]
                k = 2
                while k <= ordm1:
                    mmk = ord_ - k
                    b[mmk] = a[mmk] - p * b[mmk + 1] - q * b[mmk + 2]
                    c[mmk] = b[mmk] - p * c[mmk + 1] - q * c[mmk + 2]
                    if b[mmk] > lim or c[mmk] > lim:
                        break
                    k += 1
                if k > ordm1:                  # synthetic division ran to completion
                    b[0] = a[0] - p * b[1] - q * b[2]
                    if b[0] <= lim:
                        k += 1
                if k <= ord_:                  # a coefficient blew past lim -> restart
                    break
                err = abs(b[0]) + abs(b[1])
                if err <= _LB_MAX_ERR:
                    found = True
                    break
                den = c[2] * c[2] - c[3] * (c[1] - b[1])
                if den == 0.0:
                    break
                delp = (c[2] * b[1] - c[3] * b[0]) / den
                delq = (c[2] * b[0] - b[1] * (c[1] - b[1])) / den
                p += delp
                q += delq
            if found:
                break
            # didn't converge: jump to random starting values, like the C source
            p = float(np.random.rand() - 0.5)
            q = float(np.random.rand() - 0.5)

        r1r, r1i, r2r, r2i, ok = _qquad(1.0, p, q)
        if not ok:
            return False
        rootr[ordm1], rooti[ordm1] = r1r, r1i
        rootr[ordm2], rooti[ordm2] = r2r, r2i
        # deflate: a <- quotient polynomial (degree ord-2)
        for i in range(ordm2 + 1):
            a[i] = b[i + 2]
        ord_ -= 2

    if ord_ == 2:
        r1r, r1i, r2r, r2i, ok = _qquad(a[2], a[1], a[0])
        if not ok:
            return False
        rootr[1], rooti[1] = r1r, r1i
        rootr[0], rooti[0] = r2r, r2i
        return True
    if ord_ < 1:
        return False
    # ord_ == 1: a single real root of a[1] x + a[0]
    rootr[0] = (-a[0] / a[1]) if a[1] != 0.0 else 100.0
    rooti[0] = 0.0
    return True


# converts LPC coefs into pole frequencies and bandwidths in Hz
# takes LPC coefs as input
def formant(lpc_ord: int, s_freq: float, lpca: np.ndarray,
            init: bool, rr: np.ndarray, ri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # highest possible frequency
    nyquist = s_freq / 2.0
    # constant that converts radians to Hz
    pi2t    = 2.0 * np.pi / s_freq

    if USE_LBPOLY:
        # Lin-Bairstow path. lbpoly treats lpca as a polynomial in INCREASING
        # power (index = power), so it factors 1 + a1 x + ... + ap x^p whose roots
        # are the *reciprocals* of the synthesis poles. |theta| and |bandwidth|
        # are invariant under that reciprocal, so freq/band come out identical.
        if init:
            # seed the root search just outside the unit circle (radius 2), spread
            # in angle, so the warm start has something sane on the first frame.
            x = np.pi / (lpc_ord + 1)
            for i in range(lpc_ord + 1):
                flo = lpc_ord - i
                rr[i] = 2.0 * np.cos((flo + 0.5) * x)
                ri[i] = 2.0 * np.sin((flo + 0.5) * x)
        a = [float(v) for v in lpca]          # copy; lbpoly deflates in place
        if not _lbpoly(a, lpc_ord, rr, ri):
            return np.empty(0), np.empty(0)

        freq_list = []
        band_list = []
        ii = 0
        while ii < lpc_ord:
            rri = rr[ii]
            rii = ri[ii]
            if rri != 0.0 or rii != 0.0:
                theta = math.atan2(rii, rri)
                freq_list.append(abs(theta) / pi2t)
                bb = 0.5 * s_freq * math.log(rri * rri + rii * rii) / np.pi
                band_list.append(-bb if bb < 0.0 else bb)
                # complex conjugate pair is adjacent: don't emit it twice
                if (ii + 1 <= lpc_ord and rri == rr[ii + 1]
                        and rii == -ri[ii + 1] and rii != 0.0):
                    ii += 1
            ii += 1
        freq = np.array(freq_list)
        band = np.array(band_list)
    else:
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

    if len(freq) == 0:
        return freq, band

    # split poles into junk poles (frequency ≈ 0 or ≈ Nyquist) and real poles
    # sort real poles from low to high frequency
    # dump unsorted poles at the end
    is_complex = (freq > 1.0) & (freq < nyquist)
    complex_idx = np.where(is_complex)[0]
    real_idx    = np.where(~is_complex)[0]
    order       = np.concatenate([complex_idx[np.argsort(freq[complex_idx])], real_idx])
    freq, band  = freq[order], band[order]

    # Match Snack's formant(): only the genuine complex poles (1 < f < Nyquist-1)
    # are returned as formant candidates. The trailing real/junk poles stay in the
    # C array but are excluded via maxp = *n_form, so candy() never sees them.
    # Returning the full array here would let a junk pole at f≈Nyquist be offered
    # as an F4 candidate (fmins[3]=2000 <= Nyquist <= fmaxs[3]=5000), corrupting
    # the track. We truncate instead; the sorted complex poles below Nyquist-1 are
    # exactly the first n_form entries.
    n_form = int(np.sum((freq > 1.0) & (freq < nyquist - 1.0)))
    return freq[:n_form], band[:n_form]


# ---------------------------------------------------------------------------
# Stabilized-covariance LPC solve (port of dcovlpc()/dreflpc() from sigproc2.c).
# Snack does NOT solve phi*a = shi directly. It Cholesky-factors phi, forward-
# solves for c = L^-1 shi, turns those into reflection (PARCOR) coefficients,
# and rebuilds the polynomial from them via the Levinson step-up. This yields a
# guaranteed-stable, slightly different polynomial than a direct solve, and the
# difference lands on every voiced frame's poles. We reproduce it exactly.
# ---------------------------------------------------------------------------

def _dreflpc(refl: np.ndarray, n: int) -> np.ndarray:
    """Reflection (PARCOR) coefficients -> LPC polynomial a[0..n], a[0]=1.
    Levinson step-up, matching dreflpc()."""
    a = np.zeros(n + 1)
    a[0] = 1.0
    if n >= 1:
        a[1] = refl[0]
    for i in range(2, n + 1):
        k = refl[i - 1]
        a[i] = k
        lo, hi = 1, i - 1
        half = i // 2
        while lo <= half:                       # update symmetric pairs (j, i-j)
            ta1   = a[lo] + k * a[hi]
            a[hi] = a[hi] + k * a[lo]
            a[lo] = ta1
            lo += 1
            hi -= 1
    return a


def _dcovlpc(phi: np.ndarray, shi: np.ndarray, ps: float, n: int) -> np.ndarray:
    """Stabilized covariance LPC. phi: (n,n) regularized covariance, shi: (n,)
    cross-correlation, ps: regularized residual energy. Returns LPC polynomial
    coefficients [1, a1, ..., am, 0, ...] of length n+1 (zero-padded if the
    effective order m is reduced by rank loss), matching dcovlpc()."""
    thres = 1.0e-31
    # --- Cholesky factor L (lower), with reciprocal diagonal; rank detection ---
    L       = np.zeros((n, n))
    diaginv = np.zeros(n)
    m_chol  = n
    for i in range(n):
        for j in range(i + 1):
            sm = phi[i, j] - float(L[i, :j] @ L[j, :j])
            if i == j:
                if sm <= 0.0:                   # not positive-definite -> stop (dchlsky)
                    m_chol = i
                    break
                L[i, i]   = math.sqrt(sm)
                diaginv[i] = 1.0 / L[i, i]
            else:
                L[i, j] = sm * diaginv[j]
        else:
            continue
        break

    # --- forward solve L c = shi  (dlwrtrn) ---
    c = np.zeros(n)
    for i in range(m_chol):
        c[i] = (shi[i] - float(L[i, :i] @ c[:i])) * diaginv[i]

    # --- partial residual energies; a_energy[i] = sqrt(ps - sum_{k<=i} c[k]^2) ---
    aenergy = np.zeros(n)
    ee = ps
    m  = 0
    for i in range(m_chol):
        ee = ee - c[i] * c[i]
        if ee < thres:
            break
        aenergy[i] = math.sqrt(ee)
        m += 1

    # --- reflection coefficients ---
    refl = np.zeros(n)
    if m >= 1 and ps > 0.0:
        refl[0] = -c[0] / math.sqrt(ps)
    for i in range(1, m):
        refl[i] = -c[i] / aenergy[i - 1]

    # --- rebuild polynomial; zero-pad beyond the effective order m ---
    a   = _dreflpc(refl, m)
    out = np.zeros(n + 1)
    out[:m + 1] = a
    return out


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
    try:
        # unregularized solve, used only to estimate the prediction-error
        # residual "ee" that sets the regularization strength
        a0 = np.linalg.solve(phi, shi)
    except np.linalg.LinAlgError:
        return np.zeros(lpc_ord + 1), energy

    ee  = pss - float(np.dot(shi, a0))
    pre = 0.09 * ee
    # these particular weights (0.375, -0.25, 0.0625) encode a second-difference operator
    # matches Snack src smoothing
    pre3, pre2, pre0 = 0.375 * pre, 0.25 * pre, 0.0625 * pre
    for i in range(p):
        phi[i, i] += pre3
        if i > 0:
            phi[i, i - 1] -= pre2
            phi[i - 1, i] -= pre2
        if i > 1:
            phi[i, i - 2] += pre0
            phi[i - 2, i] += pre0
    shi[0] -= pre2
    shi[1] += pre0

    # Snack's stabilized-covariance solve (Cholesky -> reflection coeffs ->
    # Levinson step-up), NOT a direct matrix solve. ps is the ridge-regularized
    # residual energy (C: a[np] = pss + pre3). Returns [1, a1, ..., ap].
    lpca = _dcovlpc(phi, shi, pss + pre3, p)
    return lpca, energy


# takes raw audio and returns per-frame resonant poles and bandwidths
def lpc_poles(signal: np.ndarray, samprate: float, frame_int: float,
              lpc_ord: int) -> list[dict]:
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
    # persistent root-search state (warm start) shared across frames, matching
    # the static rr/ri arrays in Snack's formant(); reset whenever init is True.
    rr = np.zeros(lpc_ord + 1)
    ri = np.zeros(lpc_ord + 1)
    # frame loop
    for j in range(nfrm):
        start = j * step
        frame = data[start:start + size]

        lpca, energy = lpcbsa(lpc_ord, size, frame, preemp)
        # convert LPC coefficients to poles
        if energy > 1.0:
            freq, band = formant(lpc_ord, samprate, lpca, init, rr, ri)
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

    # F1/F2 merge penalty. NOTE: in Snack's jkFormant.c this is a function-scope
    # variable that is only ever *written* when k==0 and the F1 candidate is
    # non-missing; if F1 is missing for a candidate, `merger` silently retains
    # whatever value was last computed (for a previous candidate, possibly in a
    # previous frame). This is faithfully reproduced here rather than resetting
    # `merger` to 0.0 per-candidate.
    merger = 0.0

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
            for k in range(nform):
                ic = int(cands[j, k])
                if ic >= 0:
                    if k == 0:
                        ic1 = int(cands[j, 1])
                        merger = merge_cost if (domerge and freq[ic] == freq[ic1]) else 0.0
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

    # Values are emitted on the native frame grid (frame j -> index j). The
    # window-center alignment shift (snack reports frame-start indices, so each
    # value really belongs at j + window/2 ms) is applied centrally in
    # registry.run() via WINDOW_MS, so it is not duplicated per module.
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


