# wav loading

import io
import soundfile as sf
from pathlib import Path


def read_wav(wavfile, channel=0):
    wavfile = Path(wavfile)
    if wavfile.suffix.lower() == ".mp3":
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(wavfile)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        y, fs = sf.read(buf, dtype="float64", always_2d=True)
    else:
        y, fs = sf.read(wavfile, dtype="float64", always_2d=True)
    return y[:, channel], fs
#y is used later as what the functions extract features from 
#y and fs are also used to cacluate datalen in api.py (datalen = number of frames in the audio file)

#helper that normalizes a directory of wav files into a single list of wav paths
def resolve_paths(source):
    if isinstance(source, list):
        return [Path(p) for p in source]
    source = Path(source)
    if source.is_dir():
        wavs = sorted(source.glob("*.wav"))
        mp3s = sorted(source.glob("*.mp3"))
        return wavs + mp3s
    return [source]