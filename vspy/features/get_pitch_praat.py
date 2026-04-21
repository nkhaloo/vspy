# Pitch detection using Praat. 
# Praat uses filtered autocorrelation which low passes the signal, then looks for local maxima in a distance function that compares how similar the signal is to a time shifter (lag) copy of itself. 

import numpy as np 
import parselmouth 

# define a function that prepares a wavfile to be inputed to praat 
# sets hyperparamter defaults 
def get_pitch_praat(
    wavfile,
    frameshift_ms=1,
    datalen=None,
    min_f0=75,
    max_f0=600,
    sil_threshold=0.03,
    voicing_threshold=0.45,
    octave_cost=0.01,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14
):
    #load soundfile into parselmouth 
    snd = parselmouth.Sound(str(wavfile))
    #Praat uses s not ms, so we convert framshift to s
    frameshift_s = frameshift_ms / 1000
    #run pitch tracker using hyperparams 
    pitch = snd.to_pitch_ac(
            time_step=frameshift_s,
            pitch_floor=min_f0,
            pitch_ceiling=max_f0,
            silence_threshold=sil_threshold,
            voicing_threshold=voicing_threshold,
            octave_cost=octave_cost,
            octave_jump_cost=octave_jump_cost,
            voiced_unvoiced_cost=voiced_unvoiced_cost,
        )
    # returns an array of timestaps: 1 value for each frame in the pitch track
    times = pitch.xs()
    #extract frequency (Hz)
    f0_values = pitch.selected_array["frequency"]

    # allocate output array with NaN (unvoiced frames)
    # datalen is passed in by the api call which calculates the length of the audio
    f0 = np.full(datalen, np.nan)

    # convert times to ms indices and fill output array
    for t, f in zip(times, f0_values):
        i = round(t * 1000 / frameshift_ms)
        if 0 <= i < datalen:
            f0[i] = f if f > 0 else np.nan  # praat returns 0 for unvoiced

    return f0

    
    

