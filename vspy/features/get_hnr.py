import numpy as np


def get_hnr(y, Fs, F0, n_periods=3, frameshift_ms=1.0):
    """
    Harmonics-to-noise ratio via the de Krom (1993) cepstrum-based method.
    Port of VoiceSauce func_GetHNR.m / getHNR.m.
    Returns four arrays: HNR05, HNR15, HNR25, HNR35 (dB, cutoffs at 500/1500/2500/3500 Hz).
    """
    n_frames = len(F0)
    sampleshift = Fs / 1000.0 * frameshift_ms

    HNR05 = np.full(n_frames, np.nan)
    HNR15 = np.full(n_frames, np.nan)
    HNR25 = np.full(n_frames, np.nan)
    HNR35 = np.full(n_frames, np.nan)

    for k in range(n_frames):
        ks = round((k + 1) * sampleshift)

        if ks <= 0 or ks > len(y):
            continue

        f0_curr = F0[k]
        if np.isnan(f0_curr) or f0_curr == 0:
            continue

        N0_curr = Fs / f0_curr

        ystart = round(ks - n_periods / 2 * N0_curr)
        yend   = round(ks + n_periods / 2 * N0_curr) - 1
        if (yend - ystart + 1) % 2 == 0:
            yend -= 1

        if ystart <= 0 or yend > len(y):
            continue

        yseg = y[ystart - 1:yend]  
        hnr = _hnr_segment(yseg, f0_curr, Fs)

        HNR05[k] = hnr[0]
        HNR15[k] = hnr[1]
        HNR25[k] = hnr[2]
        HNR35[k] = hnr[3]

    return HNR05, HNR15, HNR25, HNR35


def _hnr_segment(y, F0, Fs, n_freqs=(500, 1500, 2500, 3500)):
    N = len(y)
    N0 = round(Fs / F0)
    N0_delta = round(N0 * 0.1)

    y = y * np.hamming(N)
    Y = np.fft.fft(y, N)
    aY = np.log10(np.abs(Y) + 1e-30)
    ay = np.fft.ifft(aY).real
    n_rahmonics = N // 2 // N0
    for k in range(1, n_rahmonics + 1):
        seg_start = max(k * N0 - N0_delta - 1, 0)
        seg_end   = min(k * N0 + N0_delta, N)
        ayseg = ay[seg_start:seg_end]
        if len(ayseg) < 3:
            continue

        peak_idx = int(np.argmax(np.abs(ayseg)))
        s = np.sign(np.diff(ayseg))
        if peak_idx > 0:
            d = np.where(s[peak_idx - 1::-1] != 1)[0]
            l_inx = (peak_idx - d[0]) if len(d) else 0
        else:
            l_inx = 0
        if peak_idx < len(s):
            d = np.where(s[peak_idx + 1:] == 1)[0]
            r_inx = (peak_idx + d[0] + 1) if len(d) else len(ayseg) - 1
        else:
            r_inx = len(ayseg) - 1

        ay[seg_start + l_inx : seg_start + r_inx + 1] = 0

   
   
    mid_l = N // 2 + 1 
    ay[mid_l:] = ay[mid_l - 1:0:-1]

   
    N_spec = np.fft.fft(ay).real
    Ha = aY - N_spec

    
    Hdelta = F0 / Fs * N
    stop = N // 2 + 1  
    f = Hdelta + 1e-4
    while f <= stop:
        fstart = int(np.ceil(f - Hdelta))       
        fend   = int(f + 0.5)                    
        seg = Ha[fstart - 1:fend]
        if len(seg) > 0:
            N_spec[fstart - 1:fend] -= abs(float(np.min(seg)))
        f += Hdelta

   
    H_spec = aY - N_spec
    hnr = np.zeros(len(n_freqs))
    for i, freq in enumerate(n_freqs):
        Ef = int(freq / Fs * N + 0.5)  
        if Ef > 0:
            hnr[i] = 20 * np.mean(H_spec[:Ef]) - 20 * np.mean(N_spec[:Ef])

    return hnr
