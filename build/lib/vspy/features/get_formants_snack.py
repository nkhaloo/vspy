import math
import numpy as np
import pandas as pd
import soundfile as sf





MAXCAN      = 300
MAXFORMANTS = 7
MAXORDER    = 30

MISSING  = 1.0


NOBAND   = 1000.0



DF_FACT  = 20.0


DFN_FACT = 0.3


BAND_FACT = 0.002


F_BIAS   = 0.000

F_MERGE  = 2000.0



fnom  = np.array([500,  1500, 2500, 3500, 4500, 5500, 6500], dtype=np.float64)


fmins = np.array([ 50,   400, 1000, 2000, 2000, 3000, 3000], dtype=np.float64)
fmaxs = np.array([1500, 3500, 4500, 5000, 6000, 6000, 8000], dtype=np.float64)

domerge = True









USE_LBPOLY = False








DITHER_SEED = 1



SNACK_PI = 3.1415927


def get_formants(signal: np.ndarray, samprate: float, frame_int: float,
                 lpc_ord: int, nform: int = 4,
                 ds_freq: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:

    if nform > (lpc_ord - 4) // 2:
        raise ValueError(
            f"nform ({nform}) must be <= (lpc_ord - 4) // 2 = {(lpc_ord - 4) // 2}"
        )
    if nform > MAXFORMANTS:
        raise ValueError(f"nform must be <= {MAXFORMANTS}")
    sig, effective_rate = preprocess(signal, samprate, ds_freq)
    poles = lpc_poles(sig, effective_rate, frame_int, lpc_ord)








    frame_rate = int(1.0 / frame_int)
    return dpform(poles, nform, frame_rate)





def _ratprx(a: float, qlim: int = 10) -> tuple[int, int]:
    aa = abs(a)
    ai = int(aa)
    af = aa - ai

    em, pp, qq = 1.0, 0, 0



    for q in range(1, qlim + 1):
        ps = q * af
        ip = int(ps + 0.5)
        e  = abs((ps - ip) / q)
        if e < em:
            em, pp, qq = e, ip, q

    k = int(ai * qq + pp)
    return (-k if a < 0 else k), qq




def _lc_lin_fir(fc: float, nf: int = 127) -> np.ndarray:
    n = (nf + 1) // 2
    coef = np.zeros(n)
    coef[0] = 2.0 * fc
    i = np.arange(1, n)
    coef[1:] = np.sin(i * 2.0 * SNACK_PI * fc) / (SNACK_PI * i)
    fn = 2.0 * SNACK_PI / (nf - 1)
    coef *= (0.5 + 0.5 * np.cos(fn * np.arange(n)))
    return coef





def _do_fir(buf: np.ndarray, ic: np.ndarray, ncoef: int, invert: bool) -> np.ndarray:
    ic = np.asarray(ic, dtype=np.int64)
    k  = 2 * ncoef - 1
    co = np.empty(k, dtype=np.int64)
    if not invert:
        co[:ncoef - 1] = ic[ncoef - 1:0:-1]
        co[ncoef - 1]  = ic[0]
        co[ncoef:]     = ic[1:ncoef]
    else:
        co[:ncoef - 1] = -ic[ncoef - 1:0:-1]
        co[ncoef - 1]  = 2 * int(ic[1:ncoef].sum())
        co[ncoef:]     = -ic[1:ncoef]

    buf = np.asarray(buf, dtype=np.int64)
    n   = len(buf)



    ext = np.concatenate([np.zeros(ncoef - 1, dtype=np.int64), buf,
                          np.zeros(ncoef, dtype=np.int64)])
    out = np.zeros(n, dtype=np.int64)
    for t in range(k):
        out += (co[t] * ext[t:t + n] + 16384) >> 15
    return out.astype(np.int16)






def _downsample_snack(sig: np.ndarray, samprate: float,
                       insert: int, decimate: int) -> tuple[np.ndarray, float]:
    ratio_t = insert / decimate
    freq2   = ratio_t * samprate


    beta = 0.5 / decimate
    b    = _lc_lin_fir(beta)


    maxi = 32767.0
    ic = np.trunc(maxi * b + 0.5).astype(np.int64)
    nz = np.nonzero(ic)[0]
    ncoef = int(nz[-1]) + 1 if len(nz) else 1
    ic = ic[:ncoef]

    in_samps = len(sig)
    imax = int(np.max(np.abs(sig))) if in_samps else 0
    if imax == 0:
        imax = 1

    if insert > 1:
        k = (32767 * 32767) // imax
    else:
        k = (16384 * 32767) // imax
    scaled = (k * sig.astype(np.int64) + 16384) >> 15

    up = np.zeros(in_samps * insert, dtype=np.int64)
    up[::insert] = scaled

    filtered = _do_fir(up, ic, ncoef, invert=False)

    out_samps = (in_samps * insert) // decimate
    out = filtered[:out_samps * decimate:decimate]

    return out.astype(np.int16), freq2



def preprocess(signal: np.ndarray, samprate: float,
               ds_freq: float = 10000.0) -> tuple[np.ndarray, int]:





    sig = np.asarray(signal)
    if sig.dtype != np.int16:
        sig = sig.astype(np.float64)
        if np.abs(sig).max() <= 1.0:
            sig = sig * 32768.0




        sig = sig.clip(-32768, 32767).round().astype(np.int16)



    if ds_freq < samprate:

        insert, decimate = _ratprx(ds_freq / samprate)
        ratio_t = insert / decimate

        if ratio_t <= 0.99:
            sig, samprate = _downsample_snack(sig, samprate, insert, decimate)



    LCSIZ = 101

    ncoef = 1 + LCSIZ // 2

    fn    = SNACK_PI * 2.0 / (LCSIZ - 1)

    scale = 32767.0 / (0.5 * LCSIZ)

    lcf   = (scale * (0.5 + 0.4 * np.cos(fn * np.arange(ncoef)))).astype(np.int16)



    out = _do_fir(sig, lcf, ncoef, invert=True)

    return out, int(samprate)




def get_fcand(freq: np.ndarray, nform: int) -> np.ndarray:

    freq  = np.asarray(freq, dtype=np.float64)

    npole = len(freq)

    pc    = np.full((MAXCAN, nform), -1, dtype=np.int16)

    ncan  = 0


    def canbe(pnumb, fnumb):
        return fmins[fnumb] <= freq[pnumb] <= fmaxs[fnumb]





    def candy(cand, pnumb, fnumb):
        nonlocal ncan

        if fnumb < nform:
            pc[cand, fnumb] = -1

        if pnumb < npole and fnumb < nform:
            if canbe(pnumb, fnumb):
                pc[cand, fnumb] = pnumb




                if domerge and fnumb == 0 and canbe(pnumb, fnumb + 1):
                    ncan += 1
                    pc[ncan, 0] = pc[cand, 0]
                    candy(ncan, pnumb, fnumb + 1)
                candy(cand, pnumb + 1, fnumb + 1)



                if (pnumb + 1) < npole and canbe(pnumb + 1, fnumb):
                    ncan += 1
                    pc[ncan, :fnumb] = pc[cand, :fnumb]
                    candy(ncan, pnumb + 1, fnumb)


            else:
                candy(cand, pnumb + 1, fnumb)


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

    return pc[:ncan + 1]








_LB_MAX_ITS  = 100
_LB_MAX_TRYS = 100
_LB_MAX_ERR  = 1.0e-6
_LB_DBL_MAX  = 1.7976931348623157e308
_LB_LIM0     = 0.5 * math.sqrt(_LB_DBL_MAX)


def _qquad(a: float, b: float, c: float):
    if a == 0.0:
        if b == 0.0:
            return 0.0, 0.0, 0.0, 0.0, False
        return -c / b, 0.0, 0.0, 0.0, True
    numi = b * b - 4.0 * a * c
    if numi >= 0.0:

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
    b = [0.0] * (order + 1)
    c = [0.0] * (order + 1)
    ord_ = order
    while ord_ > 2:
        ordm1 = ord_ - 1
        ordm2 = ord_ - 2

        if abs(rootr[ordm1]) < 1.0e-10:
            rootr[ordm1] = 0.0
        if abs(rooti[ordm1]) < 1.0e-10:
            rooti[ordm1] = 0.0
        p = -2.0 * rootr[ordm1]
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
                if k > ordm1:
                    b[0] = a[0] - p * b[1] - q * b[2]
                    if b[0] <= lim:
                        k += 1
                if k <= ord_:
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

            p = float(np.random.rand() - 0.5)
            q = float(np.random.rand() - 0.5)

        r1r, r1i, r2r, r2i, ok = _qquad(1.0, p, q)
        if not ok:
            return False
        rootr[ordm1], rooti[ordm1] = r1r, r1i
        rootr[ordm2], rooti[ordm2] = r2r, r2i

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

    rootr[0] = (-a[0] / a[1]) if a[1] != 0.0 else 100.0
    rooti[0] = 0.0
    return True




def formant(lpc_ord: int, s_freq: float, lpca: np.ndarray,
            init: bool, rr: np.ndarray, ri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    nyquist = s_freq / 2.0

    pi2t    = 2.0 * np.pi / s_freq

    if USE_LBPOLY:




        if init:


            x = np.pi / (lpc_ord + 1)
            for i in range(lpc_ord + 1):
                flo = lpc_ord - i
                rr[i] = 2.0 * np.cos((flo + 0.5) * x)
                ri[i] = 2.0 * np.sin((flo + 0.5) * x)
        a = [float(v) for v in lpca]
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

                if (ii + 1 <= lpc_ord and rri == rr[ii + 1]
                        and rii == -ri[ii + 1] and rii != 0.0):
                    ii += 1
            ii += 1
        freq = np.array(freq_list)
        band = np.array(band_list)
    else:

        roots = np.roots(lpca)

        freq_list = []
        band_list = []
        for r in roots:

            if r.imag < 0:
                continue

            if r.real == 0.0 and r.imag == 0.0:
                continue

            theta = np.arctan2(r.imag, r.real)
            f = abs(theta) / pi2t

            b = abs(0.5 * s_freq * np.log(r.real**2 + r.imag**2) / np.pi)

            freq_list.append(f)
            band_list.append(b)

        freq = np.array(freq_list)
        band = np.array(band_list)

    if len(freq) == 0:
        return freq, band




    is_complex = (freq > 1.0) & (freq < nyquist)
    complex_idx = np.where(is_complex)[0]
    real_idx    = np.where(~is_complex)[0]
    order       = np.concatenate([complex_idx[np.argsort(freq[complex_idx])], real_idx])
    freq, band  = freq[order], band[order]








    n_form = int(np.sum((freq > 1.0) & (freq < nyquist - 1.0)))
    return freq[:n_form], band[:n_form]











def _dreflpc(refl: np.ndarray, n: int) -> np.ndarray:
    a = np.zeros(n + 1)
    a[0] = 1.0
    if n >= 1:
        a[1] = refl[0]
    for i in range(2, n + 1):
        k = refl[i - 1]
        a[i] = k
        lo, hi = 1, i - 1
        half = i // 2
        while lo <= half:
            ta1   = a[lo] + k * a[hi]
            a[hi] = a[hi] + k * a[lo]
            a[lo] = ta1
            lo += 1
            hi -= 1
    return a


def _dcovlpc(phi: np.ndarray, shi: np.ndarray, ps: float, n: int) -> np.ndarray:
    thres = 1.0e-31

    L       = np.zeros((n, n))
    diaginv = np.zeros(n)
    m_chol  = n
    for i in range(n):
        for j in range(i + 1):
            sm = phi[i, j] - float(L[i, :j] @ L[j, :j])
            if i == j:
                if sm <= 0.0:
                    m_chol = i
                    break
                L[i, i]   = math.sqrt(sm)
                diaginv[i] = 1.0 / L[i, i]
            else:
                L[i, j] = sm * diaginv[j]
        else:
            continue
        break


    c = np.zeros(n)
    for i in range(m_chol):
        c[i] = (shi[i] - float(L[i, :i] @ c[:i])) * diaginv[i]


    aenergy = np.zeros(n)
    ee = ps
    m  = 0
    for i in range(m_chol):
        ee = ee - c[i] * c[i]
        if ee < thres:
            break
        aenergy[i] = math.sqrt(ee)
        m += 1


    refl = np.zeros(n)
    if m >= 1 and ps > 0.0:
        refl[0] = -c[0] / math.sqrt(ps)
    for i in range(1, m):
        refl[i] = -c[i] / aenergy[i - 1]


    a   = _dreflpc(refl, m)
    out = np.zeros(n + 1)
    out[:m + 1] = a
    return out







def lpcbsa(lpc_ord: int, wind: int, data: np.ndarray, preemp: float, rng):
    owind = wind



    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(owind) / owind)


    total = owind + lpc_ord + 1







    sig = np.zeros(total, dtype=np.float64)
    n = min(len(data), total)
    sig[:n] = data[:n].astype(np.float64)
    sig += 0.016 * rng.random(total) - 0.008


    sig[:-1] = sig[1:] - preemp * sig[:-1]
    sig = sig[:owind + lpc_ord]


    energy = float(np.sqrt(np.dot(sig[lpc_ord:lpc_ord + owind],
                                  sig[lpc_ord:lpc_ord + owind]) / owind))
    if energy <= 0.0:
        return np.zeros(lpc_ord + 1), 0.0
    sig /= energy



    p   = lpc_ord

    phi = np.zeros((p, p))

    shi = np.zeros(p)

    y   = sig[p:p + owind]

    for i in range(p):

        si = sig[p - i - 1: p - i - 1 + owind]
        for j in range(i + 1):

            sj = sig[p - j - 1: p - j - 1 + owind]
            v = float(np.dot(si * sj, w))
            phi[i, j] = phi[j, i] = v

        shi[i] = float(np.dot(y * si, w))



    pss = float(np.dot(y ** 2, w))
    try:


        a0 = np.linalg.solve(phi, shi)
    except np.linalg.LinAlgError:
        return np.zeros(lpc_ord + 1), energy

    ee  = pss - float(np.dot(shi, a0))
    pre = 0.09 * ee


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




    lpca = _dcovlpc(phi, shi, pss + pre3, p)
    return lpca, energy



def lpc_poles(signal: np.ndarray, samprate: float, frame_int: float,
              lpc_ord: int) -> list[dict]:

    if lpc_ord > MAXORDER or lpc_ord < 2:
        raise ValueError(f"lpc_ord must be between 2 and {MAXORDER}")

    wdur   = 0.025
    preemp = math.exp(-62.831853 * 90.0 / samprate)

    size = round(wdur * samprate)
    step = round(frame_int * samprate)

    nfrm = 1 + int((len(signal) / samprate - size / samprate) / (step / samprate))
    if nfrm < 1:
        raise ValueError("Signal too short for given window/frame parameters")

    data  = signal.astype(np.int16)
    poles = []
    init  = True

    rng   = np.random.default_rng(DITHER_SEED)


    rr = np.zeros(lpc_ord + 1)
    ri = np.zeros(lpc_ord + 1)

    for j in range(nfrm):
        start = j * step

        frame = data[start:start + size + lpc_ord + 1]

        lpca, energy = lpcbsa(lpc_ord, size, frame, preemp, rng)

        if energy > 1.0:
            freq, band = formant(lpc_ord, samprate, lpca, init, rr, ri)
            init = False
        else:
            freq = np.empty(0)
            band = np.empty(0)
            init = True

        poles.append({'rms': energy, 'freq': freq, 'band': band, 'npoles': len(freq)})

    return poles



def get_stat_max(rms: np.ndarray) -> float:
    return float(np.max(rms))



def dpform(poles: list[dict], nform: int, samprate: float):
    nframes = len(poles)


    dffact     = DF_FACT   * 0.01 * samprate
    bfact      = BAND_FACT / (0.01 * samprate)
    ffact      = DFN_FACT  / (0.01 * samprate)
    FBIAS      = F_BIAS    / (0.01 * samprate)
    merge_cost = F_MERGE
    global domerge
    if merge_cost > 1000.0:
        domerge = False

    rmsmax = get_stat_max(np.array([p['rms'] for p in poles]))







    merger = 0.0


    fr = np.zeros((nform, nframes))
    ba = np.zeros((nform, nframes))


    fl: list[dict] = [{'ncand': 0, 'cand': np.empty((0, nform), dtype=np.int16),
                        'cumerr': np.empty(0), 'prept': np.empty(0, dtype=np.int32),
                        'freq': np.empty(0), 'band': np.empty(0)}
                       for _ in range(nframes)]




    for i, pole in enumerate(poles):
        freq = pole['freq']
        band = pole['band']


        rmsdffact = (pole['rms'] / rmsmax) * dffact

        if len(freq) > 0:
            cands = get_fcand(freq, nform)
        else:
            cands = np.empty((0, nform), dtype=np.int16)

        ncand  = len(cands)
        prept  = np.full(ncand, -1, dtype=np.int32)
        cumerr = np.zeros(ncand)

        for j in range(ncand):

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
                            ft = 2.0 * abs(freq[ic] - prev['freq'][ip]) / (freq[ic] + prev['freq'][ip])
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
                        fr[j, i] = fr[j, i + 1]
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


def get_formants_snack(wavfile, frameshift_ms=1, datalen=None,
                       lpc_ord=12, nform=4, ds_freq=10000.0):


    y, fs = sf.read(str(wavfile), dtype='int16', always_2d=True)
    y = y[:, 0]

    frame_int = frameshift_ms / 1000.0

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


