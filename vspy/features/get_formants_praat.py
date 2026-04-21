import numpy as np 
import parselmouth
import pandas as pd

# define function with default hyperparamters 
def get_formants_praat(
    wavfile,
    frameshift_ms=1,
    datalen=None,
    num_formants=4,
    max_formant_freq=5500,
    window_length=0.025,
    #high frequency boost parameter - flattens spectral tilt, makes formant peaks easier to see
    pre_emphasis_from=50.0
):
    #load soundfile into parselmouth 
    snd = parselmouth.Sound(str(wavfile))
    #Praat uses s not ms, so we convert framshift to s
    frameshift_s = frameshift_ms / 1000
    #main praat call: for each frame, fit an LPC model to spectrum 
    #extracts poles (numerical approximations of sharp peaks in freqeuncy)
    # H(z) = 1/A(z): pole = where denominator hits 0 
    formant = snd.to_formant_burg(
            time_step=frameshift_s,
            max_number_of_formants = num_formants,
            maximum_formant = max_formant_freq,
            window_length = window_length,
            pre_emphasis_from = pre_emphasis_from
        )

    #create output arrays filled initially with NaN
    F1 = np.full(datalen, np.nan)
    F2 = np.full(datalen, np.nan)
    F3 = np.full(datalen, np.nan)
    F4 = np.full(datalen, np.nan)
    B1 = np.full(datalen, np.nan)
    B2 = np.full(datalen, np.nan)
    B3 = np.full(datalen, np.nan)
    B4 = np.full(datalen, np.nan)

    #frame loop: calculate formants at each timestamp 
    for i in range(datalen):
        t = i * frameshift_s
        # at each timestamp, get formant value
        F1[i] = formant.get_value_at_time(1, t)
        F2[i] = formant.get_value_at_time(2, t)
        F3[i] = formant.get_value_at_time(3, t)
        F4[i] = formant.get_value_at_time(4, t)
        #same for bandwidths 
        B1[i] = formant.get_bandwidth_at_time(1, t)
        B2[i] = formant.get_bandwidth_at_time(2, t)
        B3[i] = formant.get_bandwidth_at_time(3, t)
        B4[i] = formant.get_bandwidth_at_time(4, t)

    #define t_ms 
    t_ms = np.arange(datalen) * frameshift_ms    
    #build df: t_ms = integer ms timestamps starting at 0. 
    return pd.DataFrame({
        "t_ms": t_ms,
        "F1_praat": F1,
        "F2_praat": F2,
        "F3_praat": F3,
        "F4_praat": F4,
        "B1_praat": B1,
        "B2_praat": B2,
        "B3_praat": B3,
        "B4_praat": B4,
    })

