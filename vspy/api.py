#vs() call 
import pandas as pd
from pathlib import Path
from vspy.io import resolve_paths, read_wav
from vspy import registry
from vspy.align.textgrid import parse_textgrid, label_frames

# harmonic functions htat work on a wav array
from vspy.features.get_harmonics import get_harmonics
from vspy.features.get_formant_amplitudes import get_formant_amplitudes
from vspy.features.get_2k5k import get_2k5k
from vspy.features.spectral_tilts import compute_tilts

#define function, default f0 = praat 
def vspy(source, features=None, frameshift_ms=1, output_csv="output.csv",
         textgrid_dir=None, tier=None,
         f0_source='praat', formant_source='praat'):   
    if features is None:
        features = list(registry.REGISTRY.keys())
    #use io.resolve_paths to get a list of wav files
    paths = resolve_paths(source)

    all_frames = []
    #extract info from wav files
    for path in paths:
        #y = audio samples array: float64, shape (n_samples), values [-1,1]
        # sr = duration in s (same thing as fs)
        y, fs = read_wav(path)
        #datalen = # of frames (depending on frameshift) in any given audio file
        datalen = int(len(y) / fs * 1000 / frameshift_ms)

        #run all requested extractors, returns dict of {feature_name: array or DataFrame}
        results = registry.run(features, path, frameshift_ms, datalen)

        #start with time column, then merge each result in
        df = pd.DataFrame({"t_ms": range(datalen)})
        for name, result in results.items():
            if isinstance(result, pd.DataFrame):
                #multi-column extractor (e.g. formants): drop its own t_ms and concat sideways
                result = result.drop(columns=["t_ms"], errors="ignore")
                df = pd.concat([df, result], axis=1)
            else:
                #single-array extractor (e.g. f0): add as a named column
                df[name] = result
        # pull F0 and formant values out of the DF as numpy arrays
        # allows them to be passed to the extractors 
        f0_col  = f"f0_{f0_source}"
        f1_col  = f"F1_{formant_source}"
        f2_col  = f"F2_{formant_source}"
        f3_col  = f"F3_{formant_source}"

        F0 = df[f0_col].to_numpy()
        F1 = df[f1_col].to_numpy()
        F2 = df[f2_col].to_numpy()
        F3 = df[f3_col].to_numpy()

        # run the new extractors on the arrays and store them as new columns
        H1, H2, H4 = get_harmonics(y, fs, F0)
        A1, A2, A3 = get_formant_amplitudes(y, fs, F0, F1, F2, F3)
        H2K, F2K, H5K = get_2k5k(y, fs, F0)

        df['H1'], df['H2'], df['H4'] = H1, H2, H4
        df['A1'], df['A2'], df['A3'] = A1, A2, A3
        df['H2K'], df['F2K'], df['H5K'] = H2K, F2K, H5K

        tilts = compute_tilts(H1, H2, H4, A1, A2, A3, H2K, F2K, H5K,
                              F0, F1, F2, F3, fs)
        for col, arr in tilts.items():
            df[col] = arr

        df["filename"] = path.name

        #if there is a textgrid, then add its info to the df
        if textgrid_dir is not None and tier is not None:
            tg_path = Path(textgrid_dir) / (path.stem + ".TextGrid")
            intervals = parse_textgrid(tg_path, tier)
            df[tier] = label_frames(intervals, datalen)
            df = df[df[tier].notna()]
        all_frames.append(df)
        
        
    #write as csv 
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_csv, index=False)
    return combined



