
import math
import numpy as np
import tempfile
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly, lfilter
from vspy.io import resolve_paths, read_wav
from vspy import registry
from vspy.align.textgrid import parse_textgrid, label_frames


from vspy.features.get_harmonics import get_harmonics
from vspy.features.get_formant_amplitudes import get_formant_amplitudes
from vspy.features.get_2k5k import get_2k5k
from vspy.features.spectral_tilts import compute_tilts
from vspy.features.get_cpp import get_cpp
from vspy.features.get_soe import get_soe
from vspy.features.get_hnr import get_hnr
from vspy.features.get_energy import get_energy

TARGET_FS = 16000





SMOOTH_WIN = 20
SMOOTHED_COLUMNS = (
    'H1', 'H2', 'H4', 'A1', 'A2', 'A3', 'H2K', 'H5K',
    'H1c', 'H2c', 'H4c', 'A1c', 'A2c', 'A3c', 'H2Kc',
    'H1H2', 'H2H4', 'H1A1', 'H1A2', 'H1A3', 'H4H2K', 'H2KH5K',
    'CPP', 'HNR05', 'HNR15', 'HNR25', 'HNR35',
)


def _normalize_tiers(tier):
    if tier is None:
        return []
    if isinstance(tier, str):
        return [tier]
    tiers = list(tier)
    if not tiers or not all(isinstance(name, str) and name for name in tiers):
        raise ValueError("tier must be a non-empty string or iterable of strings")
    return list(dict.fromkeys(tiers))


def _resample(y, fs, target=TARGET_FS):
    if fs == target:
        return y, fs
    g = math.gcd(int(fs), target)
    return resample_poly(y, target // g, int(fs) // g), target


def _smooth(df, smoothwin=SMOOTH_WIN):
    if not smoothwin:
        return df
    b = [1.0 / smoothwin] * smoothwin
    for col in SMOOTHED_COLUMNS:
        if col in df.columns:
            df[col] = lfilter(b, 1.0, df[col].to_numpy(dtype=float))
    return df



def vspy(source, output_file, features=None, frameshift_ms=1,
         textgrid_dir=None, tier=None,
         f0_source='straight', formant_source='praat',
         nasality_p1search=950.0, nasality_tpnum=5, nasality_scoot=0.0):
    output_path = Path(output_file).expanduser()
    if output_path.is_dir():
        raise IsADirectoryError(
            f"output_file must include a filename, not just a directory: {output_path}"
        )
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}"
        )
    output_format = output_path.suffix.lower()
    if output_format not in {'.csv', '.parquet'}:
        raise ValueError("output_file must end in .csv or .parquet")
    tiers = _normalize_tiers(tier)
    if (textgrid_dir is None) != (not tiers):
        raise ValueError("textgrid_dir and tier must be provided together")
    if features is None:
        features = list(registry.REGISTRY.keys())
    else:
        features = list(features)
    if f"f0_{f0_source}" not in registry.REGISTRY:
        raise ValueError(f"Unknown F0 source: {f0_source}")
    # The selected driving F0 must be computed even with a feature subset.
    if f"f0_{f0_source}" not in features:
        features.append(f"f0_{f0_source}")
    paths = resolve_paths(source)

    all_frames = []
    for path in paths:
        y, fs = read_wav(path)
        straight_options = {}
        tier_intervals = {}
        nasality_options = {
            'p1search': nasality_p1search,
            'tpnum': nasality_tpnum,
            'scoot': nasality_scoot,
        }
        if tiers:
            textgrid_path = Path(textgrid_dir) / (path.stem + '.TextGrid')
            analysis_intervals = []
            for tier_name in tiers:
                tier_intervals[tier_name] = parse_textgrid(
                    textgrid_path, tier_name, round_times=False)
                analysis_intervals.extend(tier_intervals[tier_name])
            straight_options['intervals'] = analysis_intervals
            nasality_options['intervals'] = analysis_intervals

        y_16k, fs_16k = _resample(y, fs)
        datalen = int(len(y_16k) / fs_16k * 1000 / frameshift_ms)


        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        sf.write(tmp_path, y_16k, fs_16k, subtype='PCM_16')


        y_16k, fs_16k = read_wav(tmp_path)

        try:
            results = registry.run(features, tmp_path, frameshift_ms, datalen,
                                   straight_options=straight_options,
                                   spectral_wavfile=path,
                                   nasality_options=nasality_options,
                                   nasality_wavfile=path)
        finally:
            tmp_path.unlink(missing_ok=True)

        df = pd.DataFrame({"t_ms": np.arange(datalen) * frameshift_ms})
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


        _smooth(df)

        df["filename"] = path.name

        if tiers:
            labeled = np.zeros(datalen, dtype=bool)
            for tier_name in tiers:
                df[tier_name] = label_frames(
                    tier_intervals[tier_name], datalen, frameshift_ms)
                labeled |= df[tier_name].notna().to_numpy()
            df = df[labeled]
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    if output_format == '.parquet':
        combined.to_parquet(output_path, index=False)
    else:
        combined.to_csv(output_path, index=False)
    return combined
