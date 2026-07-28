"""Port of Snack's ESPS `get_f0` pitch tracker, as VoiceSauce invokes it.

VoiceSauce (func_SnackPitch.m) emits exactly

    s pitch -method esps -framelength 0.001 -windowlength 0.025 \
            -maxpitch 500 -minpitch 40

which overrides four of Snack's own defaults (jkGetF0.c:82-85): min_f0 50->40,
max_f0 550->500, wind_dur 0.0075->0.025, frame_step 0.01->0.001.  Every other
parameter keeps its C default (jkGetF0.c:74-89).

The goal here is *behavioural equivalence with Snack*, not a better tracker, so
this module reproduces the C literally -- including its quirks (truncation
toward zero in the coarse lag refinement, the 10000 floor in the fine NCCF
denominator, the +/-0.5 left over from integer rounding in the decimator, the
first-25-frame stationarity special cases, and the streaming DP's early commit
at the convergence point).

Sources of truth, cited inline as file:line:
    snack2.2.10/generic/jkGetF0.c   -- Get_f0, init_dp_f0, dp_f0, get_fast_cands,
                                       get_cand, downsample/downsamp/do_ffir,
                                       lc_lin_fir, peak, get_stationarity
    snack2.2.10/generic/sigproc.c   -- xhnwindow, xautoc, xdurbin, xa_to_aca,
                                       xitakura, wind_energy, xlpc, crossf, crossfi

Snack computes in C `float` (32-bit); this port uses float64 throughout.  That
is the one deliberate numeric deviation: it only ever moves results toward the
infinitely-precise answer, and the sibling formant port established that
float32-vs-float64 differences in this code path are ~1e-5 Hz.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import lfilter

from vspy.io import read_wav

# jkGetF0.c:715-718 -- streaming/DP buffer sizes, in seconds.
READ_SIZE = 0.2
DP_HIST = 0.5
DP_LIMIT = 1.0
# jkGetF0.c:725-726 -- stationarity window/analysis interval, in seconds.
STAT_WSIZE = 0.030
STAT_AINT = 0.020


# ---------------------------------------------------------------------------
# small C-semantics helpers
# ---------------------------------------------------------------------------

def _eround(x):
    """jkGetF0.c:1545-1550 `eround`: round half away from zero.

    numpy/Python round half to even, so this is not `round()`.
    """
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def _ctrunc(x):
    """C's `(int)` cast on a float: truncate toward zero.

    Python's builtin `int()` already truncates toward zero for floats; this
    wrapper exists so the call sites read as the C they reproduce.
    """
    return int(x)


def _snack_hanning(n):
    """sigproc.c:159-167 `xhnwindow`: 0.5 - 0.5*cos((0.5+i) * 2*pi/n).

    Note the half-sample offset -- unlike np.hanning(n) this window's endpoints
    are non-zero and it is periodic in n (not n-1).
    """
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((0.5 + i) * (2.0 * np.pi / n))


def _read_int16(wavfile):
    """Return the stored 16-bit sample values as float64, plus the rate.

    Snack reads a LIN16 file and hands `Snack_GetSoundData` the raw `short`
    values (jkSoundEdit.c:232), so every energy, threshold and the 10000 floor
    in `crossfi` is calibrated to int16 magnitudes.  soundfile scales by 1/32768
    on a float read, so int16 is read directly where possible and the float path
    multiplies by 32768 (not 32767) to recover the same integers.
    """
    wavfile = Path(wavfile)
    if wavfile.suffix.lower() != ".mp3":
        try:
            y, fs = sf.read(wavfile, dtype="int16", always_2d=True)
            return y[:, 0].astype(np.float64), fs
        except Exception:
            pass
    y, fs = read_wav(wavfile)
    y = np.asarray(y, dtype=np.float64)
    if np.abs(y).max() <= 1.0:
        y = y * 32768.0
    return np.clip(y, -32768, 32767).round(), fs


# ---------------------------------------------------------------------------
# decimation front end
# ---------------------------------------------------------------------------

def _lc_lin_fir(fc, nf):
    """jkGetF0.c:617-645 `lc_lin_fir`: sinc kernel windowed by a Hanning window
    with the half-sample offset.  Returns the full symmetric `nf`-tap filter
    (the C keeps only the half in `b` and mirrors it inside do_ffir)."""
    if nf % 2 != 1:
        nf += 1
    n = (nf + 1) // 2
    coef = np.empty(n, dtype=np.float64)
    coef[0] = 2.0 * fc
    i = np.arange(1, n, dtype=np.float64)
    coef[1:] = np.sin(i * (2.0 * np.pi * fc)) / (np.pi * i)
    # jkGetF0.c:640-642 -- coef[n-i-1] *= 0.5 - 0.5*cos(2*pi/nf * (i+0.5))
    fn = 2.0 * np.pi / nf
    win = 0.5 - 0.5 * np.cos(fn * (np.arange(n, dtype=np.float64) + 0.5))
    coef *= win[::-1]
    # do_ffir (jkGetF0.c:542-549) mirrors the half filter about coef[0].
    return np.concatenate([coef[::-1], coef[1:]]), nf


def _downsample(y, fs, decimate):
    """jkGetF0.c:395-441 `downsample` -> :491 `downsamp` -> :506 `do_ffir`.

    `do_ffir` primes its delay line so that output sample l is the inner product
    of the 2*ncoef-1 tap filter with signal[8l-40 .. 8l+40] -- i.e. the filter's
    40-sample group delay is *compensated*, and dsdata[m] is centred on input
    sample `m*decimate`.  A plain lfilter(b,1,x)[::dec] is centred on
    `m*dec - 40` instead, which mis-times the coarse NCCF by 2.5 ms.

    jkGetF0.c:597 also writes `(sum<0) ? sum-0.5 : sum+0.5` -- a rounding term
    left over from when the output buffer held integers.  It survives into the
    float buffer, so it is reproduced here.
    """
    ncoeff = _ctrunc(fs * 0.005) | 1            # jkGetF0.c:414
    beta = 0.5 / decimate                       # jkGetF0.c:415
    b, ncoeff = _lc_lin_fir(beta, ncoeff)
    half = (ncoeff // 2)                        # ncoefft-1; the group delay

    z = lfilter(b, 1.0, np.concatenate([y, np.zeros(half + decimate)]))
    ds = z[half::decimate]
    # jkGetF0.c:597
    return ds + np.where(ds < 0.0, -0.5, 0.5)


# ---------------------------------------------------------------------------
# cross-correlation
# ---------------------------------------------------------------------------

def _crossf(data, size, start, nlags):
    """sigproc.c:446-519 `crossf` -- full NCCF over `nlags` lags.

    Returns (correl, maxloc, maxval).  The reference-window mean is subtracted
    from the whole sequence, and the lagged-window energy is tracked
    incrementally with a floor of 1.0 applied *after* each update (sigproc.c:504).
    """
    total = size + start + nlags
    db = data[:total] - (data[:size].sum() / size)
    correl = np.zeros(nlags, dtype=np.float64)

    engref = float(np.dot(db[:size], db[:size]))
    if engref <= 0.0:
        return correl, 0, 0.0

    # sum_i db[0:size] . db[i+start : i+start+size], for i in [0, nlags)
    xc = np.correlate(db, db[:size], mode="valid")[start:start + nlags]

    # sliding energy of the lagged window
    sq = np.concatenate([[0.0], np.cumsum(db * db)])
    eng = sq[start + size: start + size + nlags] - sq[start: start + nlags]

    if eng.min() >= 1.0:
        engc = eng
    else:
        # sigproc.c:504-505 -- once the floor fires the recurrence no longer
        # tracks the true energy, so replay it literally.
        engc = np.empty(nlags)
        e = float(eng[0])
        for i in range(nlags):
            engc[i] = e
            e = e - db[i + start] ** 2 + db[i + start + size] ** 2
            if e < 1.0:
                e = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        correl[:] = xc / np.sqrt(engc * engref)
    correl[~np.isfinite(correl)] = 0.0

    amax = correl.max()
    if amax > 0.0:
        maxloc = int(np.argmax(correl)) + start
    else:
        amax, maxloc = 0.0, -1        # sigproc.c:499 iloc = -1, amax = 0
    return correl, maxloc, float(amax)


# sigproc.c:613 -- the 10000 floor on the fine NCCF denominator.  It penalises
# low-energy regions (with int16-scaled audio a voiced frame has engc*engref of
# order 1e12, so the floor only bites in near-silence).  Dropping it removes the
# only energy-dependent term in the fine correlation and lets silence produce
# full-height spurious peaks.
A_FACT = 10000.0


def _crossfi(data, size, start0, nlags0, nlags, locs):
    """sigproc.c:543-628 `crossfi` -- NCCF computed only in `nlags`-wide patches
    centred on each coarse candidate lag.

    Untouched lags stay 0 (sigproc.c:582-583), which matters: `get_cand` later
    scans the *whole* array as one sequence, so a patch edge next to a zero
    region is a legitimate peak.

    The C clamps `start` from below at `start0` but not from above at `stop`, so
    a patch may run past the end of `correl`/`dbdata` (sigproc.c:596-620).  Those
    writes are a genuine overrun in the C; here the array carries `nlags` of
    slack so they land harmlessly, and -- as in the C -- they still take part in
    the running `amax`.
    """
    total = size + start0 + nlags0
    slack = nlags + 1
    db = data[:total + slack] - (data[:size].sum() / size)
    correl = np.zeros(nlags0 + slack, dtype=np.float64)

    engref = float(np.dot(db[:size], db[:size]))
    if engref <= 0.0:
        return correl, 0, 0.0

    ref = db[:size]
    sq = np.concatenate([[0.0], np.cumsum(db * db)])
    amax, iloc = 0.0, -1
    half = nlags >> 1                        # sigproc.c:596

    for loc in locs:
        start = loc - half
        if start < start0:
            start = start0
        dq = start - start0

        eng = sq[start + size: start + size + nlags] - sq[start: start + nlags]
        if eng.min() < 1.0:
            # sigproc.c:611-612 -- floor applied *before* use, and the recurrence
            # continues from the floored value.
            e = float(eng[0])
            for i in range(nlags):
                if e < 1.0:
                    e = 1.0
                eng[i] = e
                e = e - db[i + start] ** 2 + db[i + start + size] ** 2
        for i in range(nlags):
            s = float(np.dot(ref, db[i + start: i + start + size]))
            t = s / np.sqrt(A_FACT + eng[i] * engref)
            correl[dq + i] = t
            if t > amax:
                amax = t
                iloc = i + start

    return correl, (iloc if iloc >= 0 else 0), amax


def _get_cand(correl, nlags, firstlag, maxval, cand_thresh):
    """jkGetF0.c:445-486 `get_cand`.

    Scans indices 1 .. nlags-3 only (the C's `lastl = nlags - 2` bound plus its
    three-deep primed window), and accepts shoulders and plateaus as well as
    strict peaks.
    """
    clip = cand_thresh * maxval
    lastl = nlags - 2
    seg = correl[:lastl + 1]
    if seg.size < 3:
        return np.empty(0), np.empty(0, dtype=int)
    q = seg[1:lastl]
    hit = (q > clip) & (q >= seg[2:lastl + 1]) & (q >= seg[0:lastl - 1])
    idx = np.nonzero(hit)[0] + 1
    return correl[idx].copy(), idx + firstlag


def _peak(y0, y1, y2):
    """jkGetF0.c:651-665 `peak`: parabolic fit through three points."""
    a = (y2 - y1) + 0.5 * (y0 - y2)
    if abs(a) > 0.000001:
        xp = (y0 - y2) / (4.0 * a)
        return xp, y1 - a * xp * xp
    return 0.0, y1


def _prune(peaks, locs, n_cands):
    """jkGetF0.c:346-364 / :373-391 -- partial bubble sort that leaves the
    `n_cands-1` largest peaks in descending order, keeping one slot free for the
    unvoiced hypothesis.  Only runs when ncand >= n_cands, so below that the
    candidates stay in ascending lag order."""
    if len(peaks) < n_cands:
        return peaks, locs
    order = sorted(range(len(peaks)), key=lambda i: -peaks[i])[:n_cands - 1]
    return peaks[order], locs[order]


# ---------------------------------------------------------------------------
# stationarity  (jkGetF0.c:1358-1539, sigproc.c:366-429)
# ---------------------------------------------------------------------------

def _durbin(r, p):
    """sigproc.c:252-280 `xdurbin`.  Returns (a[0..p-1], ex)."""
    a = np.zeros(p, dtype=np.float64)
    k = np.zeros(p, dtype=np.float64)
    e = r[0]
    if e == 0.0:
        return a, 0.0
    k[0] = -r[1] / e
    a[0] = k[0]
    e *= 1.0 - k[0] * k[0]
    for i in range(1, p):
        if e == 0.0:
            return a, 0.0
        s = -float(np.dot(a[:i], r[i:0:-1]))
        k[i] = (s - r[i + 1]) / e
        a[i] = k[i]
        b = a[:i + 1].copy()
        a[:i] += k[i] * b[i - 1::-1]
        e *= 1.0 - k[i] * k[i]
    return a, e


# sigproc.c:403 -- ffact for lpc_stabl = 30: 1/(1 + 10**(-30/20)) ~= 0.96934
_FFACT = 1.0 / (1.0 + np.exp((-30.0 / 20.0) * np.log(10.0)))
# sigproc.c:422 -- Hanning rms correction factor
_WFACT = 0.612372


def _xlpc(order, wsize, data, preemp=0.4):
    """sigproc.c:366-429 `xlpc` with type 3 (Hanning) and lpc_stabl = 30.

    Two details the C makes easy to miss:

    * `get_similarity` (jkGetF0.c:1378) passes `size-1` as the window size,
      because `xhnwindow` with preemphasis reads one extra sample:
      dwind[i] = w[i] * (data[i+1] - preemp*data[i])   (sigproc.c:171-172).
    * `xautoc` returns a *normalised* autocorrelation with r[0] == 1
      (sigproc.c:227,241).  That is load-bearing, because `xitakura` never
      multiplies its constant term by r[0] -- it starts the sum at
      `c = 1 + sum(a^2)` and adds only r[1..p] (sigproc.c:322-323).  The C's
      formulation is therefore the quadratic form sum A[i]A[j]r[|i-j|] *only
      if* r[0] == 1; feed it a raw autocorrelation and the distance is off by
      c*(r[0]-1)/err.  (Writing the full double sum instead is scale invariant
      and needs no normalisation -- but then it is not what Snack computes.)

    Returns (a, rho, err) where `a` is a[1..order] (the predictor coefficients),
    `rho` is the *stabilised* normalised autocorrelation that the C writes back
    through `ar` (sigproc.c:407-408), and `err` is the normalised residual.
    """
    w = _snack_hanning(wsize)
    dwind = w * (data[1:wsize + 1] - preemp * data[:wsize])

    # sigproc.c:215-243 xautoc
    sum0 = float(np.dot(dwind, dwind))
    r = np.zeros(order + 1, dtype=np.float64)
    r[0] = 1.0
    if sum0 == 0.0:
        rho = r
        a, er = _durbin(rho, order)
        return a, rho, er
    inv = 1.0 / sum0
    ac = np.correlate(dwind, dwind, mode="full")[wsize - 1: wsize + order]
    r[1:] = ac[1:] * inv

    # sigproc.c:400-409 stabilisation; rho[0] keeps r[0], rho[1..p] scaled.
    rho = r.copy()
    rho[1:] *= _FFACT
    a, er = _durbin(rho, order)
    return a, rho, er


def _itakura(a, rho_prev, err_prev, order):
    """sigproc.c:288-326 `xa_to_aca` + `xitakura`.

    xa_to_aca builds c = sum(A^2) and b[k] = 2*sum_i A[i]A[i+k] for A = [1, a];
    xitakura returns (c + sum_k b[k]*rho_prev[k]) / err_prev.  With rho_prev[0]
    == 1 that is exactly sum_{i,j} A[i]A[j]*rho_prev[|i-j|] / err_prev.
    """
    A = np.concatenate([[1.0], a])
    ac = np.correlate(A, A, mode="full")[order:]      # ac[k] = sum A[i]A[i+k]
    c = ac[0]
    b = 2.0 * ac[1:order + 1]
    s = c + float(np.dot(b, rho_prev[1:order + 1]))
    if err_prev == 0.0:
        return np.inf
    return s / err_prev


def _wind_energy(data, size):
    """sigproc.c:333-360 `wind_energy` with a Hanning window of length `size`
    (note: `size`, not the `size-1` that xlpc uses)."""
    f = _snack_hanning(size) * data[:size]
    return float(np.sqrt(np.dot(f, f) / size))


def _get_stationarity(y, fs, n_frames, step):
    """jkGetF0.c:1442-1539 `get_stationarity` + :1358-1405 `get_similarity`.

    Window geometry (jkGetF0.c:1453-1455): the current 30 ms window starts at
    `g*step + ind` and the previous one 20 ms (`agap`) earlier, with
    `ind = (agap - stat_wsize)/2 = -80` at 16 kHz -- both windows therefore
    start *before* the frame.

    Snack special-cases the leading frames of the very first buffer
    (jkGetF0.c:1504-1514).  Later buffers take the `mem` path
    (jkGetF0.c:1516-1530), which -- because `mem` is refilled from the real
    signal and shifted one frame step at a time -- reproduces the continuous
    computation exactly, so only global frames < 25 are special here.

    The 20-deep `windstat` stack (jkGetF0.c:1306-1354) caches the previous
    window's LPC so it is computed once per frame; with wReuse = agap/step = 20
    the cached entry is exactly the window `agap` samples earlier, so caching
    and recomputing are equivalent.  This keeps one `xlpc` per frame.
    """
    stat_wsize = _ctrunc(STAT_WSIZE * fs)         # jkGetF0.c:1454
    agap = _ctrunc(STAT_AINT * fs)                # jkGetF0.c:1453
    ind = int(agap - stat_wsize) // 2             # jkGetF0.c:1455 (exact, -80)
    order = _ctrunc(2.0 + fs / 1000.0)            # jkGetF0.c:1486

    # frames before this have p < fdata in the first buffer
    n_lead_p = -(-(agap - ind) // step) if step else 0     # ceil((agap-ind)/step)
    n_lead_q = -(-(0 - ind) // step) if step else 0        # ceil(-ind/step)

    stats = np.zeros(n_frames, dtype=np.float64)
    ratios = np.ones(n_frames, dtype=np.float64)

    cache = {}     # frame -> (rho, err, rms) of that frame's *current* window

    def _win(off):
        """480+1 samples starting at `off`, zero-padded off the ends."""
        out = np.zeros(stat_wsize + 1, dtype=np.float64)
        lo, hi = max(0, off), min(len(y), off + stat_wsize + 1)
        if hi > lo:
            out[lo - off: hi - off] = y[lo:hi]
        return out

    def _cur(g):
        if g not in cache:
            w = _win(g * step + ind)
            a, rho, err = _xlpc(order, stat_wsize - 1, w)
            cache[g] = (a, rho, err, _wind_energy(w, stat_wsize))
            for old in [k for k in cache if k < g - 25]:
                del cache[old]
        return cache[g]

    lag = agap // step        # 20 frames == agap samples
    for g in range(n_frames):
        if g < n_lead_q:
            # jkGetF0.c:1511-1513 -- no window at all yet.
            stats[g] = 0.01 * 0.2
            ratios[g] = 1.0
            continue
        a, rho, err, rms3 = _cur(g)
        if g < n_lead_p:
            # jkGetF0.c:1506-1509 -- init=1: get_similarity returns 0.2/10.0
            # and declares no amplitude change (jkGetF0.c:1398-1401).
            stats[g] = 0.2 / 10.0
            ratios[g] = 1.0
            continue
        _, rho1, err1, rms1 = _cur(g - lag)
        t = _itakura(a, rho1, err1, order) - 0.8      # jkGetF0.c:1390
        stats[g] = 0.2 / t if t != 0.0 else np.inf    # jkGetF0.c:1404
        if rms1 > 0.0:
            ratios[g] = (0.001 + rms3) / rms1
        elif rms3 > 0.0:
            ratios[g] = 2.0
        else:
            ratios[g] = 1.0
    return stats, ratios


# ---------------------------------------------------------------------------
# candidate generation  (jkGetF0.c:312-392 get_fast_cands)
# ---------------------------------------------------------------------------

def _get_fast_cands(y, ds, g, step, size, dec, start, nlags, n_cands,
                    lag_weight, cand_thresh):
    """jkGetF0.c:312-392.  Returns (peaks, locs, maxval, triplets).

    `triplets` holds, per surviving candidate, the (correl[j-1], correl[j],
    correl[j+1]) read out of the *fine* correlation array -- that is what the
    backtracking interpolation in dp_f0 (jkGetF0.c:1216-1233) uses, and the
    neighbours may be zeros from an uncomputed lag rather than real correlation.
    """
    lag_wt = lag_weight / nlags                 # jkGetF0.c:322
    decnlags = 1 + (nlags // dec)               # jkGetF0.c:323
    decstart = max(1, start // dec)             # jkGetF0.c:324
    decind = (g * step) // dec                  # jkGetF0.c:325
    decsize = 1 + (size // dec)                 # jkGetF0.c:326

    corp, _, maxval = _crossf(ds[decind:], decsize, decstart, decnlags)
    peaks, locs = _get_cand(corp, decnlags, decstart, maxval, cand_thresh)

    # jkGetF0.c:339-344 -- refine each coarse peak, map it to the full rate and
    # apply the lag-dependent weight.
    if len(locs):
        locs = locs.astype(np.int64)
        newlocs = np.empty(len(locs), dtype=np.int64)
        newpeaks = np.empty(len(locs), dtype=np.float64)
        for m, lag in enumerate(locs):
            j = lag - decstart - 1
            xp, yp = _peak(corp[j], corp[j + 1], corp[j + 2])
            # jkGetF0.c:342 -- `(int)(0.5 + xp*dec)` truncates toward zero, so
            # for xp*dec < -0.5 it is NOT round(): e.g. -2.7 -> -2, not -3.
            nl = int(lag) * dec + _ctrunc(0.5 + xp * dec)
            newlocs[m] = nl
            newpeaks[m] = yp * (1.0 - lag_wt * nl)
        peaks, locs = _prune(newpeaks, newlocs, n_cands)

    # jkGetF0.c:365-366 -- fine pass, 7 lags per candidate.
    corf, _, maxval = _crossfi(y[g * step:], size, start, nlags, 7, locs)
    peaks, locs = _get_cand(corf, nlags, start, maxval, cand_thresh)
    peaks, locs = _prune(peaks, locs, n_cands)

    j = locs - start
    trip = np.stack([corf[j - 1], corf[j], corf[j + 1]], axis=1) if len(j) \
        else np.empty((0, 3))
    return peaks, locs, maxval, trip


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def get_pitch_snack(wavfile, frameshift_ms=1, datalen=None, min_f0=40, max_f0=500,
                    wind_dur=0.025, n_cands=20, lag_weight=0.3, voice_bias=0.0,
                    trans_cost=0.005, trans_amp=0.5, trans_spec=0.5,
                    freq_weight=0.02, double_cost=0.35, cand_thresh=0.3):
    y, fs = _read_int16(wavfile)
    freq = float(fs)
    N = len(y)

    frame_step = frameshift_ms / 1000.0
    # jkGetF0.c:292 -- frame_step is snapped to an exact number of samples.
    frame_step = float(_ctrunc(0.5 + freq * frame_step)) / freq

    # --- init_dp_f0 (jkGetF0.c:796-855) ---
    step = _eround(frame_step * freq)
    size = _eround(wind_dur * freq)
    frame_int = step / freq
    start = _eround(freq / max_f0)
    stop = _eround(freq / min_f0)
    nlags = stop - start + 1
    ncomp = size + stop + 1
    ln2 = np.log(2.0)
    size_frame_hist = _ctrunc(DP_HIST / frame_int)
    size_frame_out = _ctrunc(DP_LIMIT / frame_int)
    lagwt = lag_weight / stop                    # jkGetF0.c:817
    freqwt = freq_weight / frame_int             # jkGetF0.c:820

    i = _ctrunc(READ_SIZE * freq)
    nfr_full = ((i - ncomp) // step) + 1 if ncomp >= step else i // step
    downpatch = (_ctrunc(freq * 0.005) + 1) // 2          # jkGetF0.c:847
    stat_wsize = _ctrunc(STAT_WSIZE * freq)
    agap = _ctrunc(STAT_AINT * freq)
    ind = int(agap - stat_wsize) // 2
    pad = downpatch + max(stat_wsize + ind, ncomp)        # jkGetF0.c:853
    buff_size = nfr_full * step + pad
    sdstep = nfr_full * step

    # --- Get_f0's read loop (jkGetF0.c:197-245): how many whole DP frames the
    # tracker actually emits, and where the buffer boundaries fall. ---
    if N < ((frame_step * 2.0) + wind_dur) * freq:        # jkGetF0.c:180
        return _emit(np.zeros(0), datalen)
    total_samps = N
    if buff_size > total_samps:
        buff_size = total_samps
    actsize = min(buff_size, N)
    ndone, chunks, n_frames = 0, [], 0
    while True:
        done = (actsize < buff_size) or (total_samps == buff_size)
        nfr = 0 if actsize < pad else (actsize - pad) // step   # get_Nframes
        chunks.append((n_frames, nfr, done))
        n_frames += nfr
        if done:
            break
        ndone += sdstep
        actsize = min(buff_size, N - ndone)
        total_samps -= sdstep
        if actsize > total_samps:
            actsize = total_samps
    if n_frames == 0:
        return _emit(np.zeros(0), datalen)

    decimate = _ctrunc(freq / 2000.0)            # jkGetF0.c:955
    if decimate <= 1:
        ds = y
    else:
        ds = _downsample(y, freq, decimate)
    # pad so the last frame's longest lag is addressable
    need_ds = ((n_frames - 1) * step) // max(decimate, 1) + 2 + \
        (1 + size // max(decimate, 1)) + (1 + nlags // max(decimate, 1)) + 2
    if len(ds) < need_ds:
        ds = np.concatenate([ds, np.zeros(need_ds - len(ds))])
    need_y = (n_frames - 1) * step + ncomp + 16
    if len(y) < need_y:
        y = np.concatenate([y, np.zeros(need_y - len(y))])

    stats, ratios = _get_stationarity(y, freq, n_frames, step)

    # --- main DP loop (jkGetF0.c:983-1111) ---
    F_locs, F_dpvals, F_prept, F_trip, F_ncands = [], [], [], [], []
    for g in range(n_frames):
        peaks, locs, maxval, trip = _get_fast_cands(
            y, ds, g, step, size, decimate, start, nlags, n_cands,
            lag_weight, cand_thresh)
        ncand = len(locs)

        # jkGetF0.c:1029-1031, 1037-1040
        mpvals = np.empty(ncand + 1, dtype=np.float64)
        if ncand:
            mpvals[:ncand] = 1.0 - peaks * (1.0 - locs * lagwt)
        mpvals[ncand] = voice_bias + maxval
        cur_locs = np.empty(ncand + 1, dtype=np.int64)
        cur_locs[:ncand] = locs
        cur_locs[ncand] = -1                      # the unvoiced hypothesis
        ncand += 1

        if g == 0:
            # jkGetF0.c:1094-1096
            F_dpvals.append(mpvals.copy())
            F_prept.append(np.zeros(ncand, dtype=np.int64))
        else:
            prev_locs = F_locs[-1]
            prev_dp = F_dpvals[-1]
            prev_voiced = prev_locs > 0
            # jkGetF0.c:1057-1065 voiced -> voiced
            lr_v = np.log(prev_locs[prev_voiced].astype(np.float64))
            # jkGetF0.c:1067 unvoiced -> voiced
            uv_to_v = trans_cost + trans_spec * stats[g] + trans_amp / ratios[g]
            # jkGetF0.c:1081 voiced -> unvoiced
            v_to_uv = trans_cost + trans_spec * stats[g] + trans_amp * ratios[g]

            dpvals = np.empty(ncand, dtype=np.float64)
            prept = np.empty(ncand, dtype=np.int64)
            for k in range(ncand):
                loc2 = cur_locs[k]
                if loc2 > 0:
                    ferr = np.empty(len(prev_locs))
                    if len(lr_v):
                        ftemp = np.log(loc2) - lr_v
                        ttemp = np.abs(ftemp)
                        ttemp = np.minimum(ttemp, double_cost + np.abs(ftemp + ln2))
                        ttemp = np.minimum(ttemp, double_cost + np.abs(ftemp - ln2))
                        ferr[prev_voiced] = ttemp * freqwt
                    ferr[~prev_voiced] = uv_to_v
                else:
                    ferr = np.where(prev_voiced, v_to_uv, 0.0)
                err = ferr + prev_dp
                minloc = int(np.argmin(err))      # strict `<`: first wins
                dpvals[k] = err[minloc] + mpvals[k]
                prept[k] = minloc
            F_dpvals.append(dpvals)
            F_prept.append(prept)

        F_locs.append(cur_locs)
        F_trip.append(trip)
        F_ncands.append(ncand)

    raw = _dp_output(chunks, F_locs, F_dpvals, F_prept, F_trip,
                     freq, start, stop, size_frame_hist, size_frame_out)
    return _emit(raw, datalen)


def _dp_output(chunks, F_locs, F_dpvals, F_prept, F_trip, freq, start, stop,
               size_frame_hist, size_frame_out):
    """jkGetF0.c:1118-1251 -- the streaming output stage.

    Snack does *not* backtrack from the last frame of the utterance.  Once
    `DP_HIST` (0.5 s) of frames have accumulated it walks every surviving path
    back from the newest frame until they all agree on one candidate, commits
    everything from there back to the tail, and can never revise it.  Only the
    final buffer backtracks from the newest frame outright.
    """
    n = len(F_locs)
    f0 = np.zeros(n, dtype=np.float64)
    tail = 0
    num_active = 0

    for base, nfr, done in chunks:
        if nfr == 0 and not done:
            continue
        head = base + nfr - 1
        num_active += nfr
        if head < 0:
            continue
        if not (num_active >= size_frame_hist or done):
            continue

        num_paths = len(F_locs[head])             # headF->dp->ncands
        best_cand = int(np.argmin(F_dpvals[head]))
        pcands = list(F_prept[head][:num_paths])
        checkpath_done = 1
        cmpth = None

        if done:
            cmpth = head                          # jkGetF0.c:1147
        else:
            frm = head
            while True:
                frm -= 1
                checkpath_done = 1
                for k in range(1, num_paths):
                    if pcands[0] != pcands[k]:
                        checkpath_done = 0
                        break
                if not checkpath_done:
                    pr = F_prept[frm]
                    pcands = [int(pr[c]) for c in pcands]
                else:
                    cmpth = frm
                    best_cand = pcands[0]
                    break
                if frm == tail:                   # jkGetF0.c:1172-1183
                    if num_active < size_frame_out:
                        checkpath_done = 0
                        cmpth = None
                    else:
                        checkpath_done = 1
                        cmpth = head
                    break

        if not checkpath_done or cmpth is None:
            continue

        i = 0
        frm = cmpth
        while frm != tail - 1:
            cand = best_cand
            loc1 = int(F_locs[frm][cand])
            best_cand = int(F_prept[frm][cand])   # jkGetF0.c:1214
            if loc1 > 0:
                ftemp = float(loc1)
                # jkGetF0.c:1217-1232.  Note the C's `cprev` is correl[j+1] and
                # its `cnext` is correl[j-1] -- the names are swapped relative
                # to lag order.  The resulting offset is algebraically
                # (correl[j-1] - correl[j+1]) / den.
                if start < loc1 < stop:
                    y0, y1, y2 = F_trip[frm][cand]
                    cormax, cprev, cnext = y1, y2, y0
                    den = 2.0 * (cprev + cnext - 2.0 * cormax)
                    if abs(den) > 0.000001:
                        ftemp += 2.0 - ((5.0 * cprev + 3.0 * cnext
                                         - 8.0 * cormax) / den)
                f0[frm] = freq / ftemp if ftemp != 0.0 else 0.0
            else:
                f0[frm] = 0.0                     # jkGetF0.c:1235
            frm -= 1
            i += 1
        tail = cmpth + 1
        num_active -= i
    return f0


def _emit(raw, datalen):
    """Snack writes 0 for an unvoiced frame (jkGetF0.c:1235); VoiceSauce then
    pads the tail with NaN (vs_ParameterEstimation.m:687):

        sF0 = [sF0; ones(data_len-length(sF0), 1)*NaN];

    so `sF0` in the reference holds 0 inside the tracked range and NaN beyond
    it.  Match both.  The leading NaN pad is applied centrally by
    registry.OFFSET_FRAMES['f0_snack'].
    """
    if datalen is None:
        datalen = len(raw)
    out = np.full(datalen, np.nan, dtype=np.float64)
    m = min(datalen, len(raw))
    out[:m] = raw[:m]
    return out
