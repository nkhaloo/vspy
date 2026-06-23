import numpy as np
from scipy.signal import find_peaks

# define main function
def get_cpp(y, fs, f0, n_periods=3, frame_shift=1.0):
    # furn f0 into number of samples 
    sample_shift = fs / 1000 * frame_shift
    n_ms = round(fs / 1000)
    # create fixed-length array of length F0 and fill with NAN 
    cpp = np.full(len(f0), np.nan)
    #frame loop
    for k in range(len(f0)):
        # find center sample of frame k 
        ks = round((k + 1) * sample_shift)  

        if ks <= 0 or ks > len(y):
            continue
        # gets current F0 
        f0_curr = f0[k]
        if np.isnan(f0_curr):
            continue
        # current window size
        n0_curr = fs / f0_curr
        # pick the start and end indices of waveform
        ystart = round(ks - n_periods / 2 * n0_curr)
        yend = round(ks + n_periods / 2 * n0_curr)  

        if ystasrt <= 0:
            continue
        if yend > len(y):
            continue
        # total segment
        yseg = y[ystart - 1 : yend - 1] 
        # fit hamming window
        yseg = yseg * np.hamming(len(yseg))
        # compute spectrum
        yseg_fft = np.fft.fft(yseg)
        # compute log of cepstrum
        yseg_c = np.fft.ifft(np.log(np.abs(yseg_fft))).real
        # convert to  dB 
        yseg_c_db = 10 * np.log10(yseg_c ** 2)
        # keeps only first half of cepstrum 
        half = len(yseg) // 2
        yseg_c_db = yseg_c_db[:half]

        if n_ms >= len(yseg_c_db):
            continue
        # find local maxima in cepstrum
        # ignore first 1ms
        search_region = yseg_c_db[n_ms:]
        peaks, _ = find_peaks(search_region)

        if len(peaks) == 0:
            continue

        # pick the peak closest to n0_curr 
        pinx = np.argmin(np.abs(peaks - (n0_curr - n_ms)))
        peak_pos_global = peaks[pinx] + n_ms  
        # convert back to original index (not in search_region) 
        peak_val = yseg_c_db[peak_pos_global]
        # matching x axis for yseg_c_db 
        x = np.arange(n_ms, len(yseg_c_db))
        p = np.polyfit(x, yseg_c_db[n_ms:], 1)
        # find value of polynomial near cepstral peak 
        base_val = np.polyval(p, peak_pos_global)
        # get prominence
        cpp[k] = peak_val - base_val

    return cpp
