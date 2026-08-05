
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import lfilter

from vspy.io import read_wav


READ_SIZE = 0.2
DP_HIST = 0.5
DP_LIMIT = 1.0

STAT_WSIZE = 0.030
STAT_AINT = 0.020






def _eround(x):
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def _ctrunc(x):
    return int(x)


def _snack_hanning(n):
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((0.5 + i) * (2.0 * np.pi / n))


def _read_int16(wavfile):
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






def _lc_lin_fir(fc, nf):
    if nf % 2 != 1:
        nf += 1
    n = (nf + 1) // 2
    coef = np.empty(n, dtype=np.float64)
    coef[0] = 2.0 * fc
    i = np.arange(1, n, dtype=np.float64)
    coef[1:] = np.sin(i * (2.0 * np.pi * fc)) / (np.pi * i)

    fn = 2.0 * np.pi / nf
    win = 0.5 - 0.5 * np.cos(fn * (np.arange(n, dtype=np.float64) + 0.5))
    coef *= win[::-1]

    return np.concatenate([coef[::-1], coef[1:]]), nf


def _downsample(y, fs, decimate):
    ncoeff = _ctrunc(fs * 0.005) | 1
    beta = 0.5 / decimate
    b, ncoeff = _lc_lin_fir(beta, ncoeff)
    half = (ncoeff // 2)

    z = lfilter(b, 1.0, np.concatenate([y, np.zeros(half + decimate)]))
    ds = z[half::decimate]

    return ds + np.where(ds < 0.0, -0.5, 0.5)






def _crossf(data, size, start, nlags):
    total = size + start + nlags
    db = data[:total] - (data[:size].sum() / size)
    correl = np.zeros(nlags, dtype=np.float64)

    engref = float(np.dot(db[:size], db[:size]))
    if engref <= 0.0:
        return correl, 0, 0.0


    xc = np.correlate(db, db[:size], mode="valid")[start:start + nlags]


    sq = np.concatenate([[0.0], np.cumsum(db * db)])
    eng = sq[start + size: start + size + nlags] - sq[start: start + nlags]

    if eng.min() >= 1.0:
        engc = eng
    else:


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
        amax, maxloc = 0.0, -1
    return correl, maxloc, float(amax)







A_FACT = 10000.0


def _crossfi(data, size, start0, nlags0, nlags, locs):
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
    half = nlags >> 1

    for loc in locs:
        start = loc - half
        if start < start0:
            start = start0
        dq = start - start0

        eng = sq[start + size: start + size + nlags] - sq[start: start + nlags]
        if eng.min() < 1.0:


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
    a = (y2 - y1) + 0.5 * (y0 - y2)
    if abs(a) > 0.000001:
        xp = (y0 - y2) / (4.0 * a)
        return xp, y1 - a * xp * xp
    return 0.0, y1


def _prune(peaks, locs, n_cands):
    if len(peaks) < n_cands:
        return peaks, locs
    order = sorted(range(len(peaks)), key=lambda i: -peaks[i])[:n_cands - 1]
    return peaks[order], locs[order]






def _durbin(r, p):
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



_FFACT = 1.0 / (1.0 + np.exp((-30.0 / 20.0) * np.log(10.0)))

_WFACT = 0.612372


def _xlpc(order, wsize, data, preemp=0.4):
    w = _snack_hanning(wsize)
    dwind = w * (data[1:wsize + 1] - preemp * data[:wsize])


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


    rho = r.copy()
    rho[1:] *= _FFACT
    a, er = _durbin(rho, order)
    return a, rho, er


def _itakura(a, rho_prev, err_prev, order):
    A = np.concatenate([[1.0], a])
    ac = np.correlate(A, A, mode="full")[order:]
    c = ac[0]
    b = 2.0 * ac[1:order + 1]
    s = c + float(np.dot(b, rho_prev[1:order + 1]))
    if err_prev == 0.0:
        return np.inf
    return s / err_prev


def _wind_energy(data, size):
    f = _snack_hanning(size) * data[:size]
    return float(np.sqrt(np.dot(f, f) / size))


def _get_stationarity(y, fs, n_frames, step):
    stat_wsize = _ctrunc(STAT_WSIZE * fs)
    agap = _ctrunc(STAT_AINT * fs)
    ind = int(agap - stat_wsize) // 2
    order = _ctrunc(2.0 + fs / 1000.0)


    n_lead_p = -(-(agap - ind) // step) if step else 0
    n_lead_q = -(-(0 - ind) // step) if step else 0

    stats = np.zeros(n_frames, dtype=np.float64)
    ratios = np.ones(n_frames, dtype=np.float64)

    cache = {}

    def _win(off):
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

    lag = agap // step
    for g in range(n_frames):
        if g < n_lead_q:

            stats[g] = 0.01 * 0.2
            ratios[g] = 1.0
            continue
        a, rho, err, rms3 = _cur(g)
        if g < n_lead_p:


            stats[g] = 0.2 / 10.0
            ratios[g] = 1.0
            continue
        _, rho1, err1, rms1 = _cur(g - lag)
        t = _itakura(a, rho1, err1, order) - 0.8
        stats[g] = 0.2 / t if t != 0.0 else np.inf
        if rms1 > 0.0:
            ratios[g] = (0.001 + rms3) / rms1
        elif rms3 > 0.0:
            ratios[g] = 2.0
        else:
            ratios[g] = 1.0
    return stats, ratios






def _get_fast_cands(y, ds, g, step, size, dec, start, nlags, n_cands,
                    lag_weight, cand_thresh):
    lag_wt = lag_weight / nlags
    decnlags = 1 + (nlags // dec)
    decstart = max(1, start // dec)
    decind = (g * step) // dec
    decsize = 1 + (size // dec)

    corp, _, maxval = _crossf(ds[decind:], decsize, decstart, decnlags)
    peaks, locs = _get_cand(corp, decnlags, decstart, maxval, cand_thresh)



    if len(locs):
        locs = locs.astype(np.int64)
        newlocs = np.empty(len(locs), dtype=np.int64)
        newpeaks = np.empty(len(locs), dtype=np.float64)
        for m, lag in enumerate(locs):
            j = lag - decstart - 1
            xp, yp = _peak(corp[j], corp[j + 1], corp[j + 2])


            nl = int(lag) * dec + _ctrunc(0.5 + xp * dec)
            newlocs[m] = nl
            newpeaks[m] = yp * (1.0 - lag_wt * nl)
        peaks, locs = _prune(newpeaks, newlocs, n_cands)


    corf, _, maxval = _crossfi(y[g * step:], size, start, nlags, 7, locs)
    peaks, locs = _get_cand(corf, nlags, start, maxval, cand_thresh)
    peaks, locs = _prune(peaks, locs, n_cands)

    j = locs - start
    trip = np.stack([corf[j - 1], corf[j], corf[j + 1]], axis=1) if len(j)\
        else np.empty((0, 3))
    return peaks, locs, maxval, trip






def get_pitch_snack(wavfile, frameshift_ms=1, datalen=None, min_f0=40, max_f0=500,
                    wind_dur=0.025, n_cands=20, lag_weight=0.3, voice_bias=0.0,
                    trans_cost=0.005, trans_amp=0.5, trans_spec=0.5,
                    freq_weight=0.02, double_cost=0.35, cand_thresh=0.3):
    y, fs = _read_int16(wavfile)
    freq = float(fs)
    N = len(y)

    frame_step = frameshift_ms / 1000.0

    frame_step = float(_ctrunc(0.5 + freq * frame_step)) / freq


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
    lagwt = lag_weight / stop
    freqwt = freq_weight / frame_int

    i = _ctrunc(READ_SIZE * freq)
    nfr_full = ((i - ncomp) // step) + 1 if ncomp >= step else i // step
    downpatch = (_ctrunc(freq * 0.005) + 1) // 2
    stat_wsize = _ctrunc(STAT_WSIZE * freq)
    agap = _ctrunc(STAT_AINT * freq)
    ind = int(agap - stat_wsize) // 2
    pad = downpatch + max(stat_wsize + ind, ncomp)
    buff_size = nfr_full * step + pad
    sdstep = nfr_full * step



    if N < ((frame_step * 2.0) + wind_dur) * freq:
        return _emit(np.zeros(0), datalen)
    total_samps = N
    if buff_size > total_samps:
        buff_size = total_samps
    actsize = min(buff_size, N)
    ndone, chunks, n_frames = 0, [], 0
    while True:
        done = (actsize < buff_size) or (total_samps == buff_size)
        nfr = 0 if actsize < pad else (actsize - pad) // step
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

    decimate = _ctrunc(freq / 2000.0)
    if decimate <= 1:
        ds = y
    else:
        ds = _downsample(y, freq, decimate)

    need_ds = ((n_frames - 1) * step) // max(decimate, 1) + 2 +\
        (1 + size // max(decimate, 1)) + (1 + nlags // max(decimate, 1)) + 2
    if len(ds) < need_ds:
        ds = np.concatenate([ds, np.zeros(need_ds - len(ds))])
    need_y = (n_frames - 1) * step + ncomp + 16
    if len(y) < need_y:
        y = np.concatenate([y, np.zeros(need_y - len(y))])

    stats, ratios = _get_stationarity(y, freq, n_frames, step)


    F_locs, F_dpvals, F_prept, F_trip, F_ncands = [], [], [], [], []
    for g in range(n_frames):
        peaks, locs, maxval, trip = _get_fast_cands(
            y, ds, g, step, size, decimate, start, nlags, n_cands,
            lag_weight, cand_thresh)
        ncand = len(locs)


        mpvals = np.empty(ncand + 1, dtype=np.float64)
        if ncand:
            mpvals[:ncand] = 1.0 - peaks * (1.0 - locs * lagwt)
        mpvals[ncand] = voice_bias + maxval
        cur_locs = np.empty(ncand + 1, dtype=np.int64)
        cur_locs[:ncand] = locs
        cur_locs[ncand] = -1
        ncand += 1

        if g == 0:

            F_dpvals.append(mpvals.copy())
            F_prept.append(np.zeros(ncand, dtype=np.int64))
        else:
            prev_locs = F_locs[-1]
            prev_dp = F_dpvals[-1]
            prev_voiced = prev_locs > 0

            lr_v = np.log(prev_locs[prev_voiced].astype(np.float64))

            uv_to_v = trans_cost + trans_spec * stats[g] + trans_amp / ratios[g]

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
                minloc = int(np.argmin(err))
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

        num_paths = len(F_locs[head])
        best_cand = int(np.argmin(F_dpvals[head]))
        pcands = list(F_prept[head][:num_paths])
        checkpath_done = 1
        cmpth = None

        if done:
            cmpth = head
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
                if frm == tail:
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
            best_cand = int(F_prept[frm][cand])
            if loc1 > 0:
                ftemp = float(loc1)




                if start < loc1 < stop:
                    y0, y1, y2 = F_trip[frm][cand]
                    cormax, cprev, cnext = y1, y2, y0
                    den = 2.0 * (cprev + cnext - 2.0 * cormax)
                    if abs(den) > 0.000001:
                        ftemp += 2.0 - ((5.0 * cprev + 3.0 * cnext
                                         - 8.0 * cormax) / den)
                f0[frm] = freq / ftemp if ftemp != 0.0 else 0.0
            else:
                f0[frm] = 0.0
            frm -= 1
            i += 1
        tail = cmpth + 1
        num_active -= i
    return f0


def _emit(raw, datalen):
    if datalen is None:
        datalen = len(raw)
    out = np.full(datalen, np.nan, dtype=np.float64)
    m = min(datalen, len(raw))
    out[:m] = raw[:m]
    return out
