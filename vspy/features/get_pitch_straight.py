# Pitch detection using legacy STRAIGHT (Kawahara's fixed-point method).
#
# VoiceSauce calls TANDEM-STRAIGHT, which ships only as encrypted MATLAB P-code
# (exF0candidatesTSTRAIGHTGB.p et al.), so there is nothing to transcribe there.
# This is a port of *legacy* STRAIGHT (github.com/HidekiKawahara/legacy_STRAIGHT,
# STRAIGHTV40_007, Apache-2.0), whose source is public. That is the right target:
# func_StraightPitch.m keeps framePeriod at 1ms "to maintain consistency with old
# Straight" and applies a 9-point smoother "to make results comparable with v1.27
# and below" -- i.e. VoiceSauce is actively trying to reproduce legacy STRAIGHT.
#
# Pipeline (exstraightsource.m):
#   1. fixpF0VexMltpBG4 - Gabor wavelet bank -> instantaneous frequency -> fixed
#      points of the (center freq -> IF) map. Each fixed point is an F0 candidate
#      carrying a C/N (noise) score.
#   2. f0track5          - greedy tracker over candidates, gated on total/high-band
#      power, with backtracking at voicing onset.
#   3. refineF06         - F0-adaptive reanalysis using the first 3 harmonics.

import numpy as np
from scipy.signal import cheby1, filtfilt, firwin, fftconvolve, lfilter

# ---- MATLAB compatibility helpers -----------------------------------------


def _mround(x):
    """MATLAB round(): halves go away from zero. np.round() is banker's rounding,
    which shifts frame indices by one sample on exact .5 boundaries."""
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def _colon(start, step, stop):
    """MATLAB start:step:stop (inclusive), as a float array."""
    n = int(np.floor((stop - start) / step + 1e-12)) + 1
    return start + step * np.arange(max(n, 0))


def _hanning(n):
    """MATLAB hanning(n) -- symmetric, WITHOUT the zero endpoints that
    np.hanning(n) includes."""
    return np.hanning(n + 2)[1:-1]


def _fir1(order, cutoff, pass_zero=True):
    """MATLAB fir1(order, Wn) -> order+1 taps, Hamming, scaled. Wn is in units of
    Nyquist, which is scipy's default when fs is unspecified."""
    return firwin(order + 1, cutoff, window="hamming", pass_zero=pass_zero, scale=True)


def _fftfilt(b, x):
    """MATLAB fftfilt(): linear convolution truncated to len(x) (no group-delay
    compensation -- callers correct for it explicitly)."""
    return fftconvolve(np.asarray(x, dtype=float), np.asarray(b, dtype=float))[: len(x)]


def _decimate(x, q):
    """MATLAB decimate(): order-8 Chebyshev-I, zero-phase (filtfilt), cutoff 0.8/q."""
    q = int(q)
    if q <= 1:
        return np.asarray(x, dtype=float)
    b, a = cheby1(8, 0.05, 0.8 / q)
    return filtfilt(b, a, np.asarray(x, dtype=float))[::q]


def _gcbs(x, k):
    """Cardinal-B-spline-ish window used to normalize the C/N measure."""
    tt = np.asarray(x, dtype=float) + 0.0000001
    return tt**k * np.exp(-np.pi * tt**2) * (np.sin(np.pi * tt + 0.0001) / (np.pi * tt + 0.0001)) ** 2


def _nrmlcf2(f, scale2=True):
    """Normalization constants for the P/N map.

    NOTE: fixpF0VexMltpBG4.m and refineF06.m each define a local znrmlcf2 and they
    are NOT the same -- the fixp one multiplies both terms by 2, the refine one
    does not. Same name, different constants; keep them distinct."""
    n = 100
    x = np.arange(0, 3 + 1.0 / n / 2, 1.0 / n)
    g = _gcbs(x, 0)
    dg = np.append(np.diff(g), 0) * n
    dgs = dg / 2 / np.pi / f
    xx = 2 * np.pi * f * x
    mult = 2.0 if scale2 else 1.0
    c1 = np.sum((xx * dgs) ** 2) / n * mult
    c2 = np.sum((xx**2 * dgs) ** 2) / n * mult
    return c1, c2


# ---- stage 1: fixed-point analysis ----------------------------------------


def _cleaning_low_noise(x, fs, f0floor):
    """Highpass at f0floor, built as (1 - lowpass) via spectral inversion."""
    flp = int(_mround(fs * 50 / 1000))
    nn = len(x)
    wlp = _fir1(flp * 2, f0floor / (fs / 2))
    wlp[flp] = wlp[flp] - 1  # MATLAB wlp(flp+1)
    wlp = -wlp
    tx = np.concatenate([np.asarray(x, dtype=float).ravel(), np.zeros(2 * len(wlp))])
    ttx = _fftfilt(wlp, tx)
    return ttx[np.arange(nn) + flp]


def _multanalyt_fine_cspb(x, fs, f0floor, nvc, nvo, mu, mlt):
    """Wavelet bank of iso-metric Gabor filters (Gaussian * triangular cardinal
    spline), one per channel, tuned to harmonic `mlt`."""
    t0 = 1.0 / f0floor
    lmx = int(_mround(6 * t0 * fs * mu))
    wl = int(2 ** np.ceil(np.log(lmx) / np.log(2)))
    x = np.asarray(x, dtype=float).ravel()
    nx = len(x)
    tx = np.concatenate([x, np.zeros(wl)])
    gent = (np.arange(1, wl + 1) - wl / 2) / fs

    pm = np.zeros((nvc, nx), dtype=complex)
    mpv = 1.0
    for ii in range(nvc):
        tb = gent * mpv
        t = tb[np.abs(tb) < 3.5 * mu * t0]
        wd1 = np.exp(-np.pi * (t / t0 / mu) ** 2)
        wd2 = np.maximum(0, 1 - np.abs(t / t0 / mu))
        wd2 = wd2[wd2 > 0]
        wwd = np.convolve(wd2, wd1)
        wwd = wwd[np.abs(wwd) > 0.00001]
        wbias = int(_mround((len(wwd) - 1) / 2))
        # MATLAB: t(round((1:length(wwd))-wbias+length(t)/2))
        sel = _mround(np.arange(1, len(wwd) + 1) - wbias + len(t) / 2).astype(int) - 1
        sel = np.clip(sel, 0, len(t) - 1)
        wwd = wwd * np.exp(1j * 2 * np.pi * mlt * t[sel] / t0)
        pmtmp1 = fftconvolve(tx, wwd)[: len(tx)]
        pm[ii, :] = pmtmp1[wbias : wbias + nx] * np.sqrt(mpv)
        mpv = mpv * (2.0 ** (1.0 / nvo))
    return pm


def _wvlt2ifq(pm, fs):
    """Wavelet coefficients -> instantaneous frequency map."""
    mm = pm.shape[1]
    pm = pm / np.abs(pm)
    prev = np.column_stack([pm[:, 0], pm[:, : mm - 1]])
    pif = np.abs(pm - prev)
    pif = fs / np.pi * np.arcsin(np.clip(pif / 2, -1.0, 1.0))
    pif[:, 0] = pif[:, 1]
    return pif


def _ifq2gpm2(pif, f0floor, nvo):
    """Slope of the IF map along the channel (log-frequency) axis."""
    nn = pif.shape[0]
    fx = f0floor * 2.0 ** (np.arange(nn) / nvo) * 2 * np.pi
    c = 2.0 ** (1.0 / nvo)
    core = ((pif[1 : nn - 1, :] - pif[0 : nn - 2, :]) / (1 - 1 / c) + (pif[2:nn, :] - pif[1 : nn - 1, :]) / (c - 1)) / 2
    slp = np.vstack([core[0:1, :], core, core[nn - 3 : nn - 2, :]])
    return slp / fx[:, None]


def _smoothmapb(map_, fs, f0floor, nvo, mu, mlim, pex):
    """Bi-reciprocal smoothing of the P/N map along time, with a channel-dependent
    window length (harmonic-compensating)."""
    nvc, mm = map_.shape
    t0 = 1.0 / f0floor
    lmx = int(_mround(6 * t0 * fs * mu))
    wl = int(2 ** np.ceil(np.log(lmx) / np.log(2)))
    gent = (np.arange(1, wl + 1) - wl / 2) / fs

    smap = map_.copy()
    mpv = 1.0
    zt = np.zeros(wl)
    for ii in range(nvc):
        t = gent * mpv
        t = t[np.abs(t) < 3.5 * mu * t0]
        wbias = int(_mround((len(t) - 1) / 2))
        wd1 = np.exp(-np.pi * (t / (t0 * (1 - pex)) / mu) ** 2)
        wd2 = np.exp(-np.pi * (t / (t0 * (1 + pex)) / mu) ** 2)
        wd1 = wd1 / np.sum(wd1)
        wd2 = wd2 / np.sum(wd2)
        tm = _fftfilt(wd1, np.concatenate([map_[ii, :], zt]))
        tm = _fftfilt(wd2, np.concatenate([1.0 / tm[np.arange(mm) + wbias], zt]))
        smap[ii, :] = 1.0 / tm[np.arange(mm) + wbias]
        if t0 * mu / mpv * 1000 > mlim:
            mpv = mpv * (2.0 ** (1.0 / nvo))
    return smap


def _fixpfreq3(fxx, pif2, mmp, dfv, pm):
    """Locate fixed points: channels where the IF map crosses the identity line
    downward. Values are linearly interpolated onto the crossing."""
    aav = np.abs(pm)
    nn = len(fxx)
    cd1 = pif2 - fxx
    cd2 = np.append(np.diff(cd1), cd1[nn - 1] - cd1[nn - 2])
    cdd1 = np.append(cd1[1:nn], cd1[nn - 1])
    fp = (cd1 * cdd1 < 0) & (cd2 < 0)
    ixx = np.nonzero(fp)[0]
    if len(ixx) == 0:
        e = np.array([])
        return e, e, e, e
    denom = cd1[ixx] - cdd1[ixx]
    ff = pif2[ixx] + (pif2[ixx + 1] - pif2[ixx]) * cd1[ixx] / denom
    w = (ff - fxx[ixx]) / (fxx[ixx + 1] - fxx[ixx])
    vv = mmp[ixx] + (mmp[ixx + 1] - mmp[ixx]) * w
    df = dfv[ixx] + (dfv[ixx + 1] - dfv[ixx]) * w
    aa = aav[ixx] + (aav[ixx + 1] - aav[ixx]) * w
    return ff, vv, df, aa


def _fixp_f0_vex_mltp_bg4(x, fs, f0floor, nvc, nvo, mu, shiftm, smp, minm, pc, nc):
    """Fixed-point analysis of the frequency-to-instantaneous-frequency map.
    Returns per-frame candidate frequencies and their C/N scores."""
    x = _cleaning_low_noise(x, fs, f0floor)
    fxx = f0floor * 2.0 ** (np.arange(nvc) / nvo)
    fxh = np.max(fxx)
    dn = max(1, int(np.floor(fs / (fxh * 6.3))))

    pm2 = pif2 = pm3 = pif3 = None
    if nc > 2:
        pm3 = _multanalyt_fine_cspb(_decimate(x, dn), fs / dn, f0floor, nvc, nvo, mu, 3)
        pif3 = _wvlt2ifq(pm3, fs / dn)[:, ::3]
        pm3 = pm3[:, ::3]
    if nc > 1:
        pm2 = _multanalyt_fine_cspb(_decimate(x, dn), fs / dn, f0floor, nvc, nvo, mu, 2)
        pif2 = _wvlt2ifq(pm2, fs / dn)[:, ::3]
        pm2 = pm2[:, ::3]

    pm1 = _multanalyt_fine_cspb(_decimate(x, dn * 3), fs / (dn * 3), f0floor, nvc, nvo, mu, 1)
    # safeguard added 15/Jan./2003 -- avoid 0/0 in the IF map
    mxpm1 = np.max(np.abs(pm1))
    eeps = mxpm1 / 10000000
    pm1[pm1 == 0] = pm1[pm1 == 0] + eeps
    pif1 = _wvlt2ifq(pm1, fs / (dn * 3))

    mm = pif1.shape[1]
    if nc > 1:
        mm = min(mm, pif2.shape[1])
    if nc > 2:
        mm = min(mm, pif3.shape[1])

    # amplitude-weighted nonlinear blend of the first nc harmonic IF maps
    if nc == 1:
        pif2 = pif1
    elif nc == 2:
        a1, a2 = np.abs(pm1[:, :mm]) ** pc, np.abs(pm2[:, :mm]) ** pc
        pif2 = (pif1[:, :mm] * a1 + pif2[:, :mm] / 2 * a2) / (a1 + a2)
    else:
        a1 = np.abs(pm1[:, :mm]) ** pc
        a2 = np.abs(pm2[:, :mm]) ** pc
        a3 = np.abs(pm3[:, :mm]) ** pc
        pif2 = (pif1[:, :mm] * a1 + pif2[:, :mm] / 2 * a2 + pif3[:, :mm] / 3 * a3) / (a1 + a2 + a3)

    pif2 = pif2 * 2 * np.pi
    dn = dn * 3

    slp = _ifq2gpm2(pif2, f0floor, nvo)
    nn, mm = pif2.shape
    dpif = np.column_stack([(pif2[:, 1:mm] - pif2[:, 0 : mm - 1]) * fs / dn, np.zeros(nn)])
    dpif[:, mm - 1] = dpif[:, mm - 2]
    dslp = _ifq2gpm2(dpif, f0floor, nvo)

    pm1 = pm1[:, :mm]
    damp = np.column_stack([(np.abs(pm1[:, 1:mm]) - np.abs(pm1[:, 0 : mm - 1])) * fs / dn, np.zeros(nn)])
    damp[:, mm - 1] = damp[:, mm - 2]
    damp = damp / np.abs(pm1)

    fxx = f0floor * 2.0 ** (np.arange(nn) / nvo) * 2 * np.pi
    c1, c2b = _nrmlcf2(1, scale2=True)
    # cff is multiplied by 0 in the original -- the amplitude-derivative term
    # (added 01/May/1999) is disabled, leaving mmp = (dslp/sqrt(c2))^2+(slp/sqrt(c1))^2
    c2 = c2b * (fxx / 2 / np.pi) ** 2
    mmp = (dslp / np.sqrt(c2)[:, None]) ** 2 + (slp / np.sqrt(c1)) ** 2

    smap = _smoothmapb(mmp, fs / dn, f0floor, nvo, smp, minm, 0.4) if smp != 0 else mmp

    npmax = int(round(nn / 3))
    fixpp = np.zeros((npmax, mm))
    fixvv = np.full((npmax, mm), 100000000.0)
    fixdf = np.full((npmax, mm), 100000000.0)
    fixav = np.full((npmax, mm), 1000000000.0)
    nf = np.zeros(mm, dtype=int)
    for ii in range(mm):
        ff, vv, df, aa = _fixpfreq3(fxx, pif2[:, ii], smap[:, ii], dpif[:, ii] / 2 / np.pi, pm1[:, ii])
        kk = min(len(ff), npmax)
        if kk:
            fixpp[:kk, ii] = ff[:kk]
            fixvv[:kk, ii] = vv[:kk]
            fixdf[:kk, ii] = df[:kk]
            fixav[:kk, ii] = aa[:kk]
        nf[ii] = kk
    fixpp[fixpp == 0] = fixpp[fixpp == 0] + 1000000

    np_ = max(int(np.max(nf)), 1)
    idx = _mround(_colon(1, shiftm / dn * fs / 1000, mm)).astype(int) - 1
    idx = np.clip(idx, 0, mm - 1)
    f0v = fixpp[:np_, idx] / 2 / np.pi
    vrv = fixvv[:np_, idx]
    dfv = fixdf[:np_, idx]
    aav = fixav[:np_, idx]
    return f0v, vrv, dfv, nf[idx], aav


# ---- power envelopes used for voicing gating ------------------------------


def _plotcpower(x, fs, shiftm):
    """Total power (pw) and >3kHz power (pwh), both smoothed and decimated to the
    frame grid. f0track5 uses their ratio to veto voicing in fricative/noise."""
    flm = 8
    fl = int(_mround(flm * fs / 1000))
    w = _hanning(2 * fl + 1)
    w = w / np.sum(w)
    nn = len(x)

    flpm = 40
    flp = int(_mround(flpm * fs / 1000))
    wlp = _fir1(flp * 2, 70 / (fs / 2))
    wlp[flp] = wlp[flp] - 1
    wlp = -wlp  # highpass at 70Hz

    tx = np.concatenate([np.asarray(x, dtype=float).ravel(), np.zeros(2 * len(wlp))])
    ttx = _fftfilt(wlp, tx)
    ttx = ttx[np.arange(nn) + flp]
    tx = np.concatenate([ttx, np.zeros(2 * len(w))])

    pw = _fftfilt(w, tx**2)
    pw = pw[np.arange(nn) + fl]
    mpw = np.max(pw)
    sel = _mround(_colon(1, shiftm * fs / 1000, nn)).astype(int) - 1
    sel = np.clip(sel, 0, nn - 1)
    pw = pw[sel]
    pw[pw < mpw / 10000000] = pw[pw < mpw / 10000000] + mpw / 10000000

    b = _fir1(2 * fl + 1, [0.0001, 3000 / (fs / 2)], pass_zero=False)
    b[fl] = b[fl] - 1  # bandpass minus impulse -> keeps energy above 3kHz
    xh = _fftfilt(b, tx)
    xh = xh[np.arange(nn) + fl]
    tx = np.concatenate([xh, np.zeros(10 * len(w))])
    pwh = _fftfilt(w, tx**2)
    pwh = pwh[np.arange(nn) + fl]
    pwh = pwh[sel]
    pwh[pwh < mpw / 10000000] = pwh[pwh < mpw / 10000000] + mpw / 10000000
    return pw, pwh


# ---- stage 2: tracking ----------------------------------------------------


def _f0track5(f0v, vrv, dfv, pwt, pwh, aav, shiftm):
    """Greedy F0 tracker. Enters voicing when the best candidate is confidently
    periodic and the band is not noise-dominated, backtracks up to 100ms to catch
    the onset, and looks ahead 10ms before giving up. Deliberately primitive --
    the original calls it 'very primitive and conventional'."""
    vrv = np.sqrt(vrv)
    mm = min(vrv.shape[1], len(pwt))

    f0 = np.zeros(mm)
    irms = np.ones(mm)
    df = np.ones(mm)
    amp = np.zeros(mm)
    von = 0
    ixx = np.argmin(vrv, axis=0)
    mxvr = np.min(vrv, axis=0)

    hth = 0.12  # confident-voiced threshold (updated 01/August/1999)
    lth = 0.9  # threshold to lose confidence
    bkls = int(100 / shiftm)  # back-track length
    lals = int(10 / shiftm)  # look-ahead length
    with np.errstate(divide="ignore", invalid="ignore"):
        htr = 10 * np.log10(pwh[:mm] / pwt[:mm])
    thf0j = 0.04 * np.sqrt(shiftm)  # 4% of F0 is the jump limit
    htrth = -2.0  # was -3, mod 2002.6.3

    ii = 0
    f0ref = 0.0
    jxx = 0
    while ii < mm:
        if von == 0 and mxvr[ii] < hth and htr[ii] < htrth:
            von = 1
            f0ref = f0v[ixx[ii], ii]
            for jj in range(ii, max(0, ii - bkls) - 1, -1):
                rel = np.abs((f0v[:, jj] - f0ref) / f0ref)
                jxx = int(np.argmin(rel))
                gomi = rel[jxx] + (f0ref > 10000) + (f0v[jxx, jj] > 10000)
                if ((gomi > thf0j or vrv[jxx, jj] > lth or htr[jj] > htrth) and f0v[jxx, jj] < 1000) and htr[jj] > -18:
                    break
                if gomi > thf0j:
                    break
                f0[jj] = f0v[jxx, jj]
                irms[jj] = vrv[jxx, jj]
                df[jj] = dfv[jxx, jj]
                amp[jj] = aav[jxx, jj]
                f0ref = f0[jj]
            f0ref = f0v[ixx[ii], ii]

        if 0 < f0ref < 10000:
            rel = np.abs((f0v[:, ii] - f0ref) / f0ref)
            jxx = int(np.argmin(rel))
            gomi = rel[jxx]
        else:
            gomi = 10.0

        if von == 1 and mxvr[ii] > hth:
            for jj in range(ii, min(mm - 1, ii + lals) + 1):
                ii = jj
                rel = np.abs((f0v[:, ii] - f0ref) / f0ref)
                jxx = int(np.argmin(rel))
                gomi = rel[jxx] + (f0ref > 10000) + (f0v[jxx, ii] > 10000)
                if gomi < thf0j and (htr[ii] < htrth or f0v[jxx, ii] >= 1000):
                    f0[ii] = f0v[jxx, ii]
                    irms[ii] = vrv[jxx, ii]
                    df[ii] = dfv[jxx, ii]
                    amp[ii] = aav[jxx, ii]
                    f0ref = f0[ii]
                if gomi > thf0j or vrv[jxx, ii] > lth or (htr[ii] > htrth and f0v[jxx, ii] < 1000):
                    von = 0
                    f0ref = 0.0
                    break
        elif von == 1 and gomi < thf0j and (htr[ii] < htrth or f0v[jxx, ii] >= 1000):
            f0[ii] = f0v[jxx, ii]
            irms[ii] = vrv[jxx, ii]
            df[ii] = dfv[jxx, ii]
            amp[ii] = aav[jxx, ii]
            f0ref = f0[ii]
        else:
            von = 0
        ii += 1
    return f0, irms, df, amp


# ---- stage 3: F0-adaptive refinement --------------------------------------


def _refine_f06(x, fs, f0raw, fftl, eta, nhmx, shiftm):
    """Re-estimate F0 with an F0-adaptive window, combining the first nhmx
    harmonics weighted by their individual reliability."""
    f0raw = np.asarray(f0raw, dtype=float).ravel()
    f0i = f0raw.copy()
    f0i[f0i == 0] = f0i[f0i == 0] + 160
    fax = np.arange(fftl) / fftl * fs
    nfr = len(f0i)

    shiftl = shiftm / 1000 * fs
    x = np.concatenate([np.zeros(fftl), np.asarray(x, dtype=float).ravel(), np.zeros(fftl)])

    tt = (np.arange(1, fftl + 1) - fftl / 2) / fs
    th = np.arange(fftl) / fftl * 2 * np.pi
    rr = np.exp(-1j * th)

    f0t = 100.0
    w1 = np.maximum(0, 1 - np.abs(tt * f0t / eta))
    w1 = w1[w1 > 0]
    wg = np.exp(-np.pi * (tt * f0t / eta) ** 2)
    wgg = wg[np.abs(wg) > 0.0002]
    wo = _fftfilt(wgg, np.concatenate([w1, np.zeros(len(wgg))]))
    xo = np.arange(len(wo)) / (len(wo) - 1)
    nlo = len(wo) - 1

    R = fftl // 2 + 1
    pif = np.zeros((R, nfr))
    dpif = np.zeros((R, nfr))
    pwm = np.zeros((R, nfr))
    rms_value = np.std(x, ddof=1)

    for kk in range(nfr):
        if f0i[kk] < 40:
            f0i[kk] = 40
        f0t = f0i[kk]
        xi = _colon(0, 1 / nlo * f0t / 100, 1)
        wa = np.interp(xi, xo, wo)
        wal = len(wa)
        bb = np.arange(wal)
        bias = int(_mround(fftl - wal / 2 + kk * shiftl))
        seg = bb + bias
        if np.std(x[seg], ddof=1) * np.std(x[seg - 1], ddof=1) * np.std(x[seg + 1], ddof=1) == 0:
            x[seg] = np.random.randn(wal) * rms_value / 100000  # 26/Sept./2005
        dcl = np.mean(x[seg])
        ff0 = np.fft.fft((x[seg - 1] - dcl) * wa, fftl)
        ff1 = np.fft.fft((x[seg] - dcl) * wa, fftl)
        ff2 = np.fft.fft((x[seg + 1] - dcl) * wa, fftl)
        fd = ff2 * rr - ff1
        fd0 = ff1 * rr - ff0
        with np.errstate(divide="ignore", invalid="ignore"):
            crf = fax + (np.real(ff1) * np.imag(fd) - np.imag(ff1) * np.real(fd)) / (np.abs(ff1) ** 2) * fs / np.pi / 2
            crf0 = fax + (np.real(ff0) * np.imag(fd0) - np.imag(ff0) * np.real(fd0)) / (np.abs(ff0) ** 2) * fs / np.pi / 2
        pif[:, kk] = crf[:R] * 2 * np.pi
        dpif[:, kk] = (crf[:R] - crf0[:R]) * 2 * np.pi
        pwm[:, kk] = np.abs(ff1[:R])

    slp = (np.vstack([pif[1:R, :], pif[R - 1 : R, :]]) - pif) / (fs / fftl * 2 * np.pi)
    dslp = (np.vstack([dpif[1:R, :], dpif[R - 1 : R, :]]) - dpif) / (fs / fftl * 2 * np.pi) * fs

    c1, c2 = _nrmlcf2(shiftm, scale2=False)
    fxx = (np.arange(R) + 0.5) / fftl * fs * 2 * np.pi

    # NOTE: the original reassigns c2 inside the loop (c2 = c2*(fxx(ii)/2/pi)^2),
    # so it compounds across channels rather than being recomputed from the base
    # constant -- unlike the fixp stage, which keeps a separate c2b. This looks
    # like a bug, but it is what produces VoiceSauce's numbers, so replicate it.
    mmp = np.zeros_like(slp)
    c2_run = c2
    for ii in range(R):
        c2_run = c2_run * (fxx[ii] / 2 / np.pi) ** 2
        mmp[ii, :] = (dslp[ii, :] / np.sqrt(c2_run)) ** 2 + (slp[ii, :] / np.sqrt(c1)) ** 2

    sml = int(_mround(1.5 * fs / 1000 / 2 / shiftm)) * 2 + 1  # ~3ms, odd
    smb = int(_mround((sml - 1) / 2))
    hn = _hanning(sml)

    finite = mmp[np.isfinite(mmp)]
    pad = (np.max(finite) * 0.0000001) if finite.size else 0.0
    padded = np.hstack([mmp, np.zeros((R, sml * 2))]) + pad
    smmp = np.apply_along_axis(lambda r: _fftfilt(hn**2 / np.sum(hn**2), r), 1, padded)
    smmp[smmp == 0] = smmp[smmp == 0] + 0.0000000001
    smmp = 1.0 / np.apply_along_axis(lambda r: _fftfilt(hn / np.sum(hn), r), 1, 1.0 / smmp)
    cols = np.clip(np.arange(nfr) + sml - 2, 0, smmp.shape[1] - 1)
    smmp = smmp[:, cols]

    # power-adaptive weighting (29/July/1999)
    spwm = np.apply_along_axis(lambda r: _fftfilt(hn / np.sum(hn), r), 1, np.hstack([pwm, np.zeros((R, sml * 2))]) + 0.00001)
    spwm[spwm == 0] = spwm[spwm == 0] + 0.00000001
    spfm = np.apply_along_axis(lambda r: _fftfilt(hn / np.sum(hn), r), 1, np.hstack([pwm * pif, np.zeros((R, sml * 2))]) + 0.00001)
    spif = (spfm / spwm)[:, np.arange(nfr) + smb]

    idx = np.maximum(0, f0i / fs * fftl)
    fqv = np.zeros((nhmx, nfr))
    vvv = np.zeros((nhmx, nfr))

    # MATLAB indexes smmp/spif with column-major linear indices; ravel(order='F')
    smmp_f = smmp.ravel(order="F")
    spif_f = spif.ravel(order="F")
    lim = len(smmp_f) - 2
    iidx = np.arange(nfr) * R + 1.0
    for ii in range(1, nhmx + 1):
        iidx = idx + iidx
        base = np.clip(np.floor(iidx).astype(int), 1, lim)
        frac = iidx - np.floor(iidx)
        vvv[ii - 1, :] = (smmp_f[base - 1] + frac * (smmp_f[base] - smmp_f[base - 1])) / (ii * ii)
        fqv[ii - 1, :] = (spif_f[base - 1] + frac * (spif_f[base] - spif_f[base - 1])) / 2 / np.pi / ii

    with np.errstate(divide="ignore", invalid="ignore"):
        vvvf = 1.0 / np.sum(1.0 / vvv, axis=0)
        f0r = np.sum(fqv / np.sqrt(vvv), axis=0) / np.sum(1.0 / np.sqrt(vvv), axis=0) * (f0raw > 0)
        ecr = np.sqrt(1.0 / vvvf) * (f0raw > 0) + (f0raw <= 0)
    return np.nan_to_num(f0r), ecr


# ---- exstraightsource: the F0 half ----------------------------------------


def _straight_f0(x, fs, f0floor=40, f0ceil=500, shiftm=1.0, nvo=24, mu=1.2,
                 smp=1, minsm=5, pcf0=0.5, nh=1, fftlf0r=1024, tstretch=1.1, nhmx=3,
                 continuous=True):
    """Port of exstraightsource.m's F0 chain (aperiodicity estimation omitted).

    continuous: legacy f0track5 zeros out unvoiced frames, but VoiceSauce reads
    TANDEM's rc.f0 -- an ungated track, with voicing kept in a separate V output
    that vs_ParameterEstimation.m never saves. So strF0 has a value nearly
    everywhere. When True, frames the tracker rejected are filled with their most
    reliable fixed-point candidate to match that convention."""
    x = np.asarray(x, dtype=float).ravel()
    nvc = int(np.ceil(np.log(f0ceil / f0floor) / np.log(2) * nvo))

    f0v, vrv, dfv, _, aav = _fixp_f0_vex_mltp_bg4(x, fs, f0floor, nvc, nvo, mu, shiftm, smp, minsm, pcf0, nh)
    pwt, pwh = _plotcpower(x, fs, shiftm)
    f0raw, _, _, _ = _f0track5(f0v, vrv, dfv, pwt, pwh, aav, shiftm)

    if continuous:
        # Bridge rejected frames by log-domain interpolation between tracked
        # anchors (edges held). Substituting the raw best fixed-point candidate
        # instead injects noise-driven junk in silence -- measured 41% gross
        # error vs 9% for the gated track -- because an unvoiced frame's "best"
        # candidate is meaningless. Interpolation approximates the smooth,
        # ungated trajectory TANDEM's autoF0Tracking hands VoiceSauce.
        v = f0raw > 0
        if v.any():
            n = np.arange(len(f0raw))
            f0raw = np.exp(np.interp(n, n[v], np.log(f0raw[v])))

    dn = int(np.floor(fs / (f0ceil * 3 * 2)))  # fix by H.K. 28/Jan./2003
    f0raw, _ = _refine_f06(_decimate(x, dn), fs / dn, f0raw, fftlf0r, tstretch, nhmx, shiftm)

    f0raw[f0raw <= 0] = 0  # safeguard 31/August/2004
    f0raw[f0raw > f0ceil] = f0ceil
    return f0raw


# ---- VoiceSauce wrapper (func_StraightPitch.m) ----------------------------


def get_pitch_straight(wavfile, frameshift_ms=1, datalen=None, min_f0=40, max_f0=500,
                       maxdur=10, continuous=False):
    """STRAIGHT F0 on the shared 1ms frame grid.

    Analysis parameters mirror VoiceSauce exactly (vs_Initialize.m: minstrF0=40,
    maxstrF0=500, maxstrdur=10; func_StraightPitch.m: framePeriod=1ms, decimated
    afterwards by vs_ParameterEstimation.m) and legacy STRAIGHT's own defaults
    (nvo=24, mu=1.2, smp=1, minsm=5, pc=0.5, nc=1, fftlf0r=1024, tstretch=1.1,
    nhmx=3). None of them were tuned to fit the reference.

    continuous: VoiceSauce's strF0 has a value at nearly every frame, because it
    stores TANDEM's ungated rc.f0 and discards the separate voicing vector V.
    Legacy STRAIGHT instead zeros frames its tracker rejects. Measured over the
    154-file test set against strF0:

        gated (default)   45.8% coverage, median abs err 0.45 Hz, 3.6% gross
        continuous=True   90.0% coverage, median abs err 4.95 Hz, 21.2% gross

    Gated is the default because interpolated F0 in regions the tracker rejected
    is frequently an octave off, and every downstream feature in api.py (H1/H2,
    CPP, HNR, SoE) is computed *from* F0 -- a wrong value is worse than a NaN.
    Set continuous=True if you need strF0's dense-column semantics."""
    import soundfile as sf

    y, fs = sf.read(str(wavfile))
    if y.ndim > 1:
        y = y[:, 0]

    L = int(np.floor(len(y) / fs * 1000)) - 1

    if len(y) / fs > maxdur:
        # NOTE: func_StraightPitch.m chunks long files here, but its overlap
        # expression `y(startx - floor(F0/1000) - 10 : endx)` indexes with the
        # whole F0 *vector* -- a bug. Sequential chunks are the evident intent.
        segs = []
        step = int(np.floor(maxdur * fs))
        for start in range(0, len(y), step):
            seg = y[start : min(start + step, len(y))]
            if len(seg) < fs * 0.05:
                continue
            segs.append(_straight_f0(seg, fs, min_f0, max_f0, 1.0, continuous=continuous))
        f0 = np.concatenate(segs) if segs else np.zeros(L)
    else:
        f0 = _straight_f0(y, fs, min_f0, max_f0, 1.0, continuous=continuous)

    # 9-point smoothing, applied to the RAW (zero-containing) track, exactly as in
    # func_StraightPitch.m -- this is what makes v1.28+ "comparable with v1.27 and
    # below". Zeros participate in the average, so unvoiced frames adjacent to
    # voiced ones get pulled to small nonzero values and survive the ==0 test.
    if len(f0) > 14:
        padded = np.concatenate([np.full(9, f0[0]), f0, np.full(4, f0[-1])])
        f0_smooth = lfilter(np.ones(9) / 9, 1, padded)
        f0 = f0_smooth[13:]

    f0 = f0[:L] if len(f0) >= L else np.concatenate([f0, np.zeros(L - len(f0))])
    f0[f0 == 0] = np.nan  # Straight sometimes outputs 0 for F0

    # decimate 1ms -> frameshift_ms, then fit datalen
    if frameshift_ms != 1:
        f0 = f0[:: int(frameshift_ms)]
    if datalen is not None:
        if len(f0) > datalen:
            f0 = f0[:datalen]
        elif len(f0) < datalen:
            f0 = np.concatenate([f0, np.full(datalen - len(f0), np.nan)])
    return f0
