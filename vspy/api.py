#vs() call
import math
import tempfile
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly, lfilter
from vspy.io import resolve_paths, read_wav
from vspy import registry
from vspy.align.textgrid import parse_textgrid, label_frames

# harmonic functions that work on a wav array
from vspy.features.get_harmonics import get_harmonics
from vspy.features.get_formant_amplitudes import get_formant_amplitudes
from vspy.features.get_2k5k import get_2k5k
from vspy.features.spectral_tilts import compute_tilts
from vspy.features.get_cpp import get_cpp
from vspy.features.get_soe import get_soe
from vspy.features.get_hnr import get_hnr
from vspy.features.get_energy import get_energy

TARGET_FS = 16000

# VoiceSauce vs_Initialize.m: vars.O_smoothwinsize = 20.  func_buildMData.m
# applies `filter(ones(smoothwin,1)/smoothwin, 1, x)` -- a *causal* moving
# average -- to these measures on the way out to text/display.  F0, formants,
# bandwidths, Energy, SHR, epoch and SoE are deliberately left unsmoothed.
SMOOTH_WIN = 20
SMOOTHED_COLUMNS = (
    'H1', 'H2', 'H4', 'A1', 'A2', 'A3', 'H2K', 'H5K',
    'H1c', 'H2c', 'H4c', 'A1c', 'A2c', 'A3c', 'H2Kc',
    'H1H2', 'H2H4', 'H1A1', 'H1A2', 'H1A3', 'H4H2K', 'H2KH5K',
    'CPP', 'HNR05', 'HNR15', 'HNR25', 'HNR35',
)


def _resample(y, fs, target=TARGET_FS):
    if fs == target:
        return y, fs
    g = math.gcd(int(fs), target)
    return resample_poly(y, target // g, int(fs) // g), target


def _smooth(df, smoothwin=SMOOTH_WIN):
    """Apply VoiceSauce's output-stage moving-average filter, in place."""
    if not smoothwin:
        return df
    b = [1.0 / smoothwin] * smoothwin
    for col in SMOOTHED_COLUMNS:
        if col in df.columns:
            df[col] = lfilter(b, 1.0, df[col].to_numpy(dtype=float))
    return df


#define function, default f0 = praat
def vspy(source, features=None, frameshift_ms=1, output_csv="output.csv",
         textgrid_dir=None, tier=None,
         f0_source='praat', formant_source='praat'):
    if features is None:
        features = list(registry.REGISTRY.keys())
    paths = resolve_paths(source)

    all_frames = []
    for path in paths:
        y, fs = read_wav(path)
        # resample everything to 16kHz once, matching VoiceSauce behaviour
        y_16k, fs_16k = _resample(y, fs)
        datalen = int(len(y_16k) / fs_16k * 1000 / frameshift_ms)

        # write temp 16kHz wav so Praat/Snack extractors use the same audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        sf.write(tmp_path, y_16k, fs_16k, subtype='PCM_16')
        # VoiceSauce re-reads the resampled file it just wrote, so every
        # measure sees the 16-bit quantised signal. Match that.
        y_16k, fs_16k = read_wav(tmp_path)

        try:
            results = registry.run(features, tmp_path, frameshift_ms, datalen)
        finally:
            tmp_path.unlink(missing_ok=True)

        df = pd.DataFrame({"t_ms": range(datalen)})
        for name, result in results.items():
            if isinstance(result, pd.DataFrame):
                result = result.drop(columns=["t_ms"], errors="ignore")
                df = pd.concat([df, result], axis=1)
            else:
                df[name] = result

        f0_col = f"f0_{f0_source}"
        f1_col = f"F1_{formant_source}"
        f2_col = f"F2_{formant_source}"
        f3_col = f"F3_{formant_source}"

        has_f0       = f0_col in df.columns
        has_formants = all(c in df.columns for c in [f1_col, f2_col, f3_col])

        if has_f0 and has_formants:
            F0 = df[f0_col].to_numpy()
            F1 = df[f1_col].to_numpy()
            F2 = df[f2_col].to_numpy()
            F3 = df[f3_col].to_numpy()

            H1, H2, H4 = get_harmonics(y_16k, fs_16k, F0)
            A1, A2, A3 = get_formant_amplitudes(y_16k, fs_16k, F0, F1, F2, F3)
            H2K, F2K, H5K = get_2k5k(y_16k, fs_16k, F0)

            df['H1'], df['H2'], df['H4'] = H1, H2, H4
            df['A1'], df['A2'], df['A3'] = A1, A2, A3
            df['H2K'], df['F2K'], df['H5K'] = H2K, F2K, H5K

            tilts = compute_tilts(H1, H2, H4, A1, A2, A3, H2K, F2K, H5K,
                                  F0, F1, F2, F3, fs_16k)
            for col, arr in tilts.items():
                df[col] = arr

            df['CPP'] = get_cpp(y_16k, fs_16k, F0, frame_shift=frameshift_ms)

            df['Energy'] = get_energy(y_16k, fs_16k, F0, frame_shift=frameshift_ms)

            soe_df = get_soe(y_16k, fs_16k, F0, frameshift_ms=frameshift_ms)
            df['epoch'] = soe_df['epoch'].values
            df['SoE'] = soe_df['soe'].values

            HNR05, HNR15, HNR25, HNR35 = get_hnr(y_16k, fs_16k, F0, frameshift_ms=frameshift_ms)
            df['HNR05'] = HNR05
            df['HNR15'] = HNR15
            df['HNR25'] = HNR25
            df['HNR35'] = HNR35

        # VoiceSauce smooths on output, per file, before any textgrid slicing.
        _smooth(df)

        df["filename"] = path.name

        if textgrid_dir is not None and tier is not None:
            tg_path = Path(textgrid_dir) / (path.stem + ".TextGrid")
            intervals = parse_textgrid(tg_path, tier)
            df[tier] = label_frames(intervals, datalen)
            df = df[df[tier].notna()]
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_csv, index=False)
    return combined
