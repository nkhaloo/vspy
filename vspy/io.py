# wav loading

import soundfile as sf 
from pathlib import Path

#define a function that returns an array of the wav file and its sampling rate 
#chanel returns first channel if sound is stereo
def read_wav(wavfile, channel=0):
    # mirrors MATLAB audioread(): y = audio sample array 
    #sf.read() returns y as a 2d array with shape (n_samples, n_channels) and fs (sampling rate)
    y, fs = sf.read(wavfile, dtype="float64", always_2d=True)
    #return a 1d array of shape (n_samples), contaning waveform amplitude values for the selected channel + fs
    return y[:, channel], fs
#y is used later as what the functions extract features from 
#y and fs are also used to cacluate datalen in api.py (datalen = number of frames in the audio file)

#helper that normalizes a directory of wav files into a single list of wav paths
def resolve_paths(source):
    if isinstance(source, list):
        return [Path(p) for p in source]
    source = Path(source)
    if source.is_dir():
        return sorted(source.glob("*.wav"))
    return [source]