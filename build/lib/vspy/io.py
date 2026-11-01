

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




def resolve_paths(source):
    if isinstance(source, list):
        paths = [Path(p).expanduser() for p in source]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Input audio file does not exist: {missing[0]}")
        if not paths:
            raise ValueError("No input audio files were provided")
        return paths
    source = Path(source).expanduser()
    if source.is_dir():
        wavs = sorted(source.glob("*.wav"))
        mp3s = sorted(source.glob("*.mp3"))
        paths = wavs + mp3s
        if not paths:
            raise ValueError(f"Input directory contains no WAV or MP3 files: {source}")
        return paths
    if not source.is_file():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    return [source]
