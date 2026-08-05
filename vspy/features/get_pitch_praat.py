


import numpy as np
import parselmouth



def get_pitch_praat(
    wavfile,
    frameshift_ms=1,
    datalen=None,
    min_f0=40,
    max_f0=500,
    sil_threshold=0.03,
    voicing_threshold=0.45,
    octave_cost=0.01,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14
):

    snd = parselmouth.Sound(str(wavfile))

    frameshift_s = frameshift_ms / 1000

    pitch = snd.to_pitch_cc(
            time_step=frameshift_s,
            pitch_floor=min_f0,
            pitch_ceiling=max_f0,
            silence_threshold=sil_threshold,
            voicing_threshold=voicing_threshold,
            octave_cost=octave_cost,
            octave_jump_cost=octave_jump_cost,
            voiced_unvoiced_cost=voiced_unvoiced_cost,
        )

    times = pitch.xs()

    f0_values = pitch.selected_array["frequency"]



    f0 = np.full(datalen, np.nan)


    for t, f in zip(times, f0_values):
        i = round(t * 1000 / frameshift_ms)
        if 0 <= i < datalen:
            f0[i] = f if f > 0 else np.nan

    return f0




