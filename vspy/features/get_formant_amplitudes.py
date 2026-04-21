# generate formant aplitudes (A1, A2, A3)
import numpy as np

# main function
def get_formant_amplitudes(y, Fs, F0, F1, F2, F3, n_periods=3):
    n_frames = len(F0)
    #empty arrays 
    A1 = np.full(n_frames, np.nan)
    A2 = np.full(n_frames, np.nan)
    A3 = np.full(n_frames, np.nan)

    # uses 8192 point FFT 
    # searches for peak within +/- 10% of the Fx value and reutrns its amplitude 
    def fft_amplitude(segment, Fx):
        X = np.fft.fft(segment, 8192)
        X_db = 20 * np.log10(np.abs(X[:4096]) + 1e-10)
        fstep = Fs / 8192
        lo = max(0, int((Fx * 0.9) / fstep))
        hi = min(4095, int((Fx * 1.1) / fstep))
        return np.max(X_db[lo:hi+1])
        
    sampleshift = Fs / 1000

    for k in range(n_frames):
        f0 = F0[k]

        if np.isnan(f0) or f0 == 0:
            continue

        if np.isnan(F1[k]) or np.isnan(F2[k]) or np.isnan(F3[k]):
            continue

        ks = round(k * sampleshift)
        N0 = Fs / f0

        ystart = round(ks - (n_periods / 2) * N0)
        yend   = round(ks + (n_periods / 2) * N0)

        if ystart < 0 or yend > len(y):
            continue

        segment = y[ystart:yend]

        A1[k] = fft_amplitude(segment, F1[k])
        A2[k] = fft_amplitude(segment, F2[k])
        A3[k] = fft_amplitude(segment, F3[k])

    return A1, A2, A3