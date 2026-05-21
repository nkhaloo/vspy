import numpy as np
import pandas as pd
from scipy.signal import resample_poly, lfilter
from math import gcd
from vspy.io import read_wav

# DP cost constant
_FNOM  = [500., 1500., 2500., 3500., 4500., 5500., 6500.]
_FMINS = [ 50.,  400., 1000., 2000., 2000., 3000., 3000.]
_FMAXS = [1500., 3500., 4500., 5000., 6000., 6000., 8000.]
_MISSING   = 1.0
_NOBAND    = 1000.0
_DF_FACT   = 20.0
_DFN_FACT  = 0.3
_BAND_FACT = 0.002
_F_BIAS    = 0.0
_F_MERGE   = 2000.0


#  1. Highpass filter 

def _highpass(y):
    """Hanning FIR highpass matching SNACK's highpass() — assumes 10 kHz input."""
    LCSIZ  = 101
    n_half = 1 + LCSIZ // 2          # 51 half-filter coefficients
    fn     = np.pi * 2.0 / (LCSIZ - 1)
    scale  = 32767.0 / (0.5 * LCSIZ)

    # ic[0] = center (large), ic[50] = edge (small)
    ic = np.array([round(scale * (0.5 + 0.4 * np.cos(fn * i)))
                   for i in range(n_half)], dtype=np.float64)

    # Inverted (highpass) filter of length 2*n_half-1 = 101.
    # Off-center coefficients are negated; center = 2*sum(ic[1:]).
    # This is SNACK's do_fir() with invert=1.
    co = np.empty(2 * n_half - 1)
    co[:n_half - 1] = -ic[n_half - 1:0:-1]  # left half: -ic[50..1]
    co[n_half - 1]  =  2.0 * np.sum(ic[1:]) # center
    co[n_half:]     = -ic[1:]                # right half: -ic[1..50]

    return lfilter(co / 32768.0, 1.0, y)


#  2. Stabilized covariance LPC 

def _lpcbsa(frame, wind, order, preemp=0.7, xl=0.09):
    """Stabilized covariance LPC matching SNACK's lpcbsa() + dlpcwtd().
    Returns (lpc_coefficients, energy).  Coefficients do not include leading 1."""
    owind = wind
    # Hamming window: period = owind (note: 2π/owind, not /(owind-1))
    w = 0.54 - 0.46 * np.cos(np.arange(owind) * (2.0 * np.pi / owind))

    # Copy frame, zero-padded to wind+order+1 samples
    total = wind + order + 1
    sig   = np.zeros(total)
    n_cp  = min(len(frame), total)
    sig[:n_cp] = frame[:n_cp]

    wind1 = wind + order  # last useful index exclusive
    # Forward preemphasis: sig[i] ← sig[i+1] − preemp·sig[i], i = 0..wind1-2
    # This matches the C loop: *(psp3-1) = *psp3 - preemp * *(psp3-1)
    sig[:wind1 - 1] = sig[1:wind1] - preemp * sig[:wind1 - 1]

    energy = np.sqrt(np.sum(sig[order:wind1] ** 2) / owind)
    if energy < 1e-10:
        return np.zeros(order), energy
    sig[:wind1] /= energy

    # Weighted covariance matrix (dcwmtrx) ─────────────────────────────────
    ni, nl = order, wind1  # ni=lpc_order, nl=wind+order

    ps  = float(np.sum(sig[ni:nl] ** 2 * w))
    shi = np.array([np.dot(sig[ni:nl] * w, sig[ni - 1 - k:nl - 1 - k])
                    for k in range(order)])

    phi = np.zeros((order, order))
    for i in range(order):
        ai = ni - i - 1
        for j in range(i + 1):
            aj  = ni - j - 1
            sm  = float(np.dot(sig[ai:ai + owind] * sig[aj:aj + owind], w))
            phi[i, j] = phi[j, i] = sm

    p_diag = np.diag(phi).copy()

    # Estimate prediction error (ee = ps − shi·phi⁻¹·shi) for ridge scaling
    try:
        L   = np.linalg.cholesky(phi + np.eye(order) * 1e-12)
        c_  = np.linalg.solve(L, shi)
        ee  = max(ps - float(np.dot(c_, c_)), 1e-30)
    except np.linalg.LinAlgError:
        ee = ps * 0.01

    # Ridge regularization matching dlpcwtd():
    #   diagonal   += pre3,  ±1 off-diagonal −= pre2,  ±2 off-diagonal += pre0
    pre  = ee * xl
    pr3, pr2, pr0 = 0.375 * pre, 0.25 * pre, 0.0625 * pre

    phi_r = phi.copy()
    np.fill_diagonal(phi_r, p_diag + pr3)
    for i in range(1, order):
        phi_r[i, i - 1] -= pr2
        phi_r[i - 1, i] -= pr2
    for i in range(2, order):
        phi_r[i, i - 2] += pr0
        phi_r[i - 2, i] += pr0

    shi_r = shi.copy()
    shi_r[0] -= pr2
    if order > 1:
        shi_r[1] += pr0

    try:
        a = np.linalg.solve(phi_r, shi_r)
    except np.linalg.LinAlgError:
        return np.zeros(order), energy
    return a, energy


# ── 3. LPC → poles ───────────────────────────────────────────────────────────

def _poles_from_lpc(a, fs):
    """Convert LPC coefficients to (freq_list, bw_list) for complex poles only."""
    roots  = np.roots(np.concatenate([[1.0], a]))
    roots  = roots[np.imag(roots) > 0]
    freqs  = np.arctan2(np.imag(roots), np.real(roots)) * (fs / (2.0 * np.pi))
    bws    = -np.log(np.abs(roots)) * (fs / np.pi)
    order  = np.argsort(freqs)
    freqs, bws = freqs[order], bws[order]
    mask   = (freqs > 1.0) & (freqs < fs / 2.0 - 1.0)
    return freqs[mask].tolist(), bws[mask].tolist()


# ── 4. Candidate generation (candy / get_fcand) ───────────────────────────────

def _get_fcand(freqs, _bands, nform):
    """All pole-to-formant candidate mappings, matching SNACK's get_fcand()."""
    n_poles = len(freqs)
    MAXCAN  = 300
    pc      = [[-1] * nform for _ in range(MAXCAN)]
    ncan    = [0]

    def canbe(p, f):
        return _FMINS[f] <= freqs[p] <= _FMAXS[f]

    def candy(cand, pnumb, fnumb):
        if ncan[0] >= MAXCAN - 1:
            return
        if fnumb < nform:
            pc[cand][fnumb] = -1
        if pnumb < n_poles and fnumb < nform:
            if canbe(pnumb, fnumb):
                pc[cand][fnumb] = pnumb
                # F1/F2 merger: same pole may also be F2
                if fnumb == 0 and fnumb + 1 < nform and canbe(pnumb, fnumb + 1):
                    ncan[0] += 1
                    nc = ncan[0]
                    pc[nc][0] = pc[cand][0]
                    candy(nc, pnumb, fnumb + 1)
                candy(cand, pnumb + 1, fnumb + 1)
                # alternative assignment: try next pole for same formant
                if pnumb + 1 < n_poles and canbe(pnumb + 1, fnumb):
                    ncan[0] += 1
                    nc = ncan[0]
                    for k in range(fnumb):
                        pc[nc][k] = pc[cand][k]
                    candy(nc, pnumb + 1, fnumb)
            else:
                candy(cand, pnumb + 1, fnumb)
        # Exhausted poles for this formant: advance to next with missing slot
        if pnumb >= n_poles and fnumb < nform - 1 and pc[cand][fnumb] < 0:
            j = fnumb - 1
            while j > 0 and pc[cand][j] < 0:
                j -= 1
            i = pc[cand][j] if (fnumb > 0 and pc[cand][j] >= 0) else 0
            candy(cand, i, fnumb + 1)

    candy(0, 0, 0)
    ncan[0] += 1
    return [list(pc[i]) for i in range(ncan[0])]


# ── 5. DP formant tracker (dpform) ───────────────────────────────────────────

def _dpform(all_freqs, all_bands, all_rms, nform, frame_rate=1000.0):
    """DP formant tracking matching SNACK's dpform().
    frame_rate is the LPC analysis rate in Hz (= 1000 / frameshift_ms)."""
    n_frames = len(all_freqs)
    rmsmax   = max((r for r in all_rms if r > 0.0), default=1.0)

    # Scale cost weights to frame rate (matching dpform()'s dffact/bfact/ffact)
    dffact = _DF_FACT   * 0.01 * frame_rate
    bfact  = _BAND_FACT / (0.01 * frame_rate)
    ffact  = _DFN_FACT  / (0.01 * frame_rate)
    FBIAS  = _F_BIAS    / (0.01 * frame_rate)

    all_cands = [_get_fcand(f, None, nform) if f else []
                 for f, _ in zip(all_freqs, all_bands)]

    def local_cost(freqs, bands, mapping):
        berr = ferr = fbias = merger = 0.0
        for k in range(nform):
            ic = mapping[k]
            if ic >= 0:
                if k == 0 and len(mapping) > 1 and mapping[1] == ic:
                    merger = _F_MERGE
                berr  += bands[ic]
                ferr  += abs(freqs[ic] - _FNOM[k]) / _FNOM[k]
                fbias += freqs[ic]
            else:
                fbias += _FNOM[k]
                berr  += _NOBAND
                ferr  += _MISSING
        return FBIAS * fbias + bfact * berr + ffact * ferr + merger

    # Forward pass ─────────────────────────────────────────────────────────
    cum_errs = []
    backptrs = []

    for i in range(n_frames):
        freqs   = all_freqs[i]
        bands   = all_bands[i]
        cands   = all_cands[i]
        ncand   = len(cands)
        rmsdff  = (all_rms[i] / rmsmax) * dffact

        lc = (np.array([local_cost(freqs, bands, c) for c in cands])
              if ncand else np.array([]))

        if i == 0:
            cum_errs.append(lc.copy() if ncand else np.array([]))
            backptrs.append(np.full(ncand, -1, dtype=int))
            continue

        prev_cands = all_cands[i - 1]
        prev_ce    = cum_errs[i - 1]
        n_prev     = len(prev_cands)
        new_ce     = np.full(ncand, np.inf)
        new_bp     = np.full(ncand, -1, dtype=int)

        for j in range(ncand):
            best_cost = np.inf
            best_k    = -1
            for k in range(n_prev):
                pferr = 0.0
                for fi in range(nform):
                    ic = cands[j][fi]
                    ip = prev_cands[k][fi]
                    if ic >= 0 and ip >= 0:
                        fc, fp = freqs[ic], all_freqs[i - 1][ip]
                        t = 2.0 * abs(fc - fp) / (fc + fp)
                        pferr += t * t
                    else:
                        pferr += _MISSING
                conerr = rmsdff * pferr + prev_ce[k]
                if conerr < best_cost:
                    best_cost, best_k = conerr, k
            new_ce[j] = lc[j] + (best_cost if n_prev else 0.0)
            new_bp[j] = best_k

        cum_errs.append(new_ce)
        backptrs.append(new_bp)

    # Backtrack ─────────────────────────────────────────────────────────────
    fr = [[_FNOM[k]  for k in range(nform)] for _ in range(n_frames)]
    ba = [[_NOBAND   for k in range(nform)] for _ in range(n_frames)]
    mincan = -1

    for i in range(n_frames - 1, -1, -1):
        ce = cum_errs[i]
        if mincan < 0 and len(ce) > 0 and not np.all(np.isinf(ce)):
            mincan = int(np.argmin(ce))

        if mincan >= 0 and mincan < len(all_cands[i]):
            mapping = all_cands[i][mincan]
            for k in range(nform):
                ic = mapping[k]
                if ic >= 0:
                    fr[i][k] = all_freqs[i][ic]
                    ba[i][k] = all_bands[i][ic]
                else:
                    if i < n_frames - 1:  # replicate backwards from next frame
                        fr[i][k] = fr[i + 1][k]
                        ba[i][k] = ba[i + 1][k]
            mincan = (int(backptrs[i][mincan])
                      if mincan < len(backptrs[i]) else -1)
        else:
            for fi in range(nform):
                if i < n_frames - 1:
                    fr[i][fi] = fr[i + 1][fi]
                    ba[i][fi] = ba[i + 1][fi]
            mincan = -1

    return fr, ba


# ── 6. Main function ──────────────────────────────────────────────────────────

def get_formants_snack(
    wavfile,
    frameshift_ms = 1,
    datalen       = None,
    num_formants  = 4,
    window_ms     = 25,
    pre_emphasis  = 0.98,
    lpc_order     = 12,
    ds_freq       = 10000,
    max_bandwidth = 500,   # noqa: unused — DP tracker uses _FMINS/_FMAXS frequency bounds
):
    y, fs = read_wav(wavfile)

    # Downsample (pre-emphasis is handled per-frame inside _lpcbsa)
    g    = gcd(ds_freq, fs)
    y_ds = resample_poly(y, ds_freq // g, fs // g).astype(np.float64)

    # Highpass filter matching SNACK's highpass() (assumes ds_freq = 10000)
    y_ds = _highpass(y_ds)

    frame_step = int(round(frameshift_ms / 1000 * ds_freq))
    frame_size = int(round(window_ms    / 1000 * ds_freq))
    n_frames   = max(0, (len(y_ds) - frame_size) // frame_step)

    all_freqs = []
    all_bands = []
    all_rms   = []

    for i in range(n_frames):
        start = i * frame_step
        frame = y_ds[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        a, energy = _lpcbsa(frame, frame_size, lpc_order, preemp=pre_emphasis)
        all_rms.append(energy)
        freqs, bws = _poles_from_lpc(a, float(ds_freq))
        all_freqs.append(freqs)
        all_bands.append(bws)

    if datalen is None:
        datalen = n_frames

    frame_rate = 1000.0 / frameshift_ms
    fr, ba = _dpform(all_freqs, all_bands, all_rms, num_formants, frame_rate)

    F = [np.full(datalen, np.nan) for _ in range(num_formants)]
    B = [np.full(datalen, np.nan) for _ in range(num_formants)]

    for i in range(n_frames):
        t_ms = i * frame_step / ds_freq * 1000
        idx  = round(t_ms / frameshift_ms)
        if 0 <= idx < datalen:
            for k in range(num_formants):
                f_val = fr[i][k]
                b_val = ba[i][k]
                if f_val != _FNOM[k] or b_val != _NOBAND:
                    F[k][idx] = f_val
                    B[k][idx] = b_val

    t_ms = np.arange(datalen) * frameshift_ms
    cols = {"t_ms": t_ms}
    for k, name in enumerate(["F1", "F2", "F3", "F4"][:num_formants]):
        cols[f"{name}_snack"] = F[k]
        cols[f"B{k+1}_snack"] = B[k]
    return pd.DataFrame(cols)
