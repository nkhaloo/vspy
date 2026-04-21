# This module computes pitch using the probabalistic YIN algorithim (pYIN). 
# YIN calculates the squared distance between a given point and its lag, normalizes it, then looks for local minima. A parablic function is fit around the local minima to refine it. Then F0 is calculated using the period of the minima
# pYIN doesn't pick a single minima, works on a number of local mininam candiates, finding proabilities across different possible pitch values. It also explciitly models voiced/unvoiced


import numpy as np
import librosa
from vspy.io import read_wav

# define the function and set default hyperparams
def get_pitch_pyin(
    wavfile,
    frameshift_ms=1,
    datalen=None,
    fmin=50,
    fmax= 200,
    frame_length=25,
    # tune for voicing. higher = more willing to voice
    switch_prob=0.5
):
    # store wav file as a numpy array
    y, fs = read_wav(wavfile)
    # set the hop length to seconds
    hop_length = int(fs * frameshift_ms / 1000)
    # converts frame_length to how many audio samples fit inside one 25ms window
    n_fft = int(fs * frame_length / 1000)
    # call librosa.pyin
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=fs,
        hop_length=hop_length,
        frame_length=n_fft,
        switch_prob = switch_prob
    )
    # np.where replaces unvoiced segments with NaN
    f0 = np.where(voiced_flag, f0, np.nan)
    # copy into the fixed-length array to ensure it matches datalen
    output = np.full(datalen, np.nan)
    for i, val in enumerate(f0):
        if 0 <= i < datalen:
            output[i] = val

    return output
