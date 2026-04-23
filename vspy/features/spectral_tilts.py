# generate functions: Hawks and Miller (generates formant bandwidths) and iseli correction
# function to compute spectral tilts 

import numpy as np

# Hawks and Miller function 
# gives a validated measure of formant bandwidths (used in correction)
def hawks_miller_bw(Fn, F0):
    # S is a, F0 scaling factor 
    # bandwidth grows as F0 rises above the reference pitch (132)
    # higher F0 = wider formant bandwidths 
    # helps for more accurate bandwidth estimation 
    S = 1 + 0.25 * (F0 - 132) / 88

    # fit a polynomial below 500Hz
    C1 = np.array([165.327516, -6.73636734e-1,  1.80874446e-3,
                  -4.52201682e-6, 7.49514000e-9, -4.70219241e-12])
    # fit polynomial above 500hz 
    C2 = np.array([15.8146139,   8.10159009e-2, -9.79728215e-5,
                    5.28725064e-8, -1.07099364e-11, 7.91528509e-16])

    F = np.array([Fn**i for i in range(6)])  # shape (6, n_frames)

    mask = (Fn < 500).astype(float)  # 1.0 where Fn < 500, else 0.0

    bw = S * (C1 @ (F * mask) + C2 @ (F * (1 - mask)))
    return bw


# iseli correction: f = harmonic frequency, Fx = formant, Bx = bandwidth, Fs = sampling rate 
def iseli_correction(f, Fx, Bx, Fs):
    # convert bandwidth into a number between 0 and 1 that captures how strong the resonance is 
    # close to 1 = strong sharp peak (thinner bandwidth)
    # think of this as a pole
    r = np.exp(-np.pi * Bx / Fs)

    # convert from Hz into radians (angles)
    # how far around the circle does one sample step? 
    # frequency at exactly the Nyquist theorum corresponds to pi radians (or 180 degrees)
    omega_x = 2 * np.pi * Fx / Fs # convert formant to radians 
    omega   = 2 * np.pi * f  / Fs # convert harmonic frequency to radians

    # measure distance between formant and harmonic frequencies on a circle (not a line)
    # theres two equations (a and b) to measure each direction around a circle
    # a and b measure how much a formant amplifies the signal at the harmonic's frequency 
    a    = r**2 + 1 - 2*r*np.cos(omega_x + omega)
    b    = r**2 + 1 - 2*r*np.cos(omega_x - omega)
    # baseline where omega = 0 
    # this basically shows a baseline effect of the formant frequency
    num  = r**2 + 1 - 2*r*np.cos(omega_x)

    # measures how much dB the nearest formant adds at this given frequency
    # just subracting a from b, the result would be "how much does this filter amplify at f?"
    # after normalization, (subtracting 20 * num), the result is 'how much ectra doers this filter amplify f, relative to its onw baseline?'
    corr = -10*(np.log10(a) + np.log10(b)) + 20*np.log10(num)
    return corr



# compute tilts 
def compute_tilts(H1, H2, H4, A1, A2, A3, H2K, F2K, H5K,
                  F0, F1, F2, F3, Fs, bandwidth='formula',
                  B1=None, B2=None, B3=None):
                      
    if bandwidth == 'formula':
        bw1 = hawks_miller_bw(F1, F0)
        bw2 = hawks_miller_bw(F2, F0)
        bw3 = hawks_miller_bw(F3, F0)
    else:
        bw1, bw2, bw3 = B1, B2, B3

    # correct each harmonic by nearest formants and formant bandwidths
    H1c = H1 - iseli_correction(F0,     F1, bw1, Fs) \
             - iseli_correction(F0,     F2, bw2, Fs)
    H2c = H2 - iseli_correction(2*F0,   F1, bw1, Fs) \
             - iseli_correction(2*F0,   F2, bw2, Fs)
    H4c = H4 - iseli_correction(4*F0,   F1, bw1, Fs) \
             - iseli_correction(4*F0,   F2, bw2, Fs)

    A1c = A1 - iseli_correction(F1, F1, bw1, Fs) \
             - iseli_correction(F1, F2, bw2, Fs)
    A2c = A2 - iseli_correction(F2, F1, bw1, Fs) \
             - iseli_correction(F2, F2, bw2, Fs)
    A3c = A3 - iseli_correction(F3, F1, bw1, Fs) \
             - iseli_correction(F3, F2, bw2, Fs) \
             - iseli_correction(F3, F3, bw3, Fs)

    H2Kc = H2K - iseli_correction(F2K, F1, bw1, Fs) \
               - iseli_correction(F2K, F2, bw2, Fs) \
               - iseli_correction(F2K, F3, bw3, Fs)

    return {
        'H1c':    H1c,
        'H2c':    H2c,
        'H4c':    H4c,
        'A1c':    A1c,
        'A2c':    A2c,
        'A3c':    A3c,
        'H2Kc':   H2Kc,
        'H1H2':   H1c  - H2c,
        'H2H4':   H2c  - H4c,
        'H1A1':   H1c  - A1c,
        'H1A2':   H1c  - A2c,
        'H1A3':   H1c  - A3c,
        'H4H2K':  H4c  - H2Kc,
        'H2KH5K': H2Kc - H5K,
    }
    