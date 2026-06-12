# registry for feature names and extraction functions
# used by api call to map features to their functions and call them
import numpy as np
import pandas as pd

from vspy.features.get_formants_snack import get_formants_snack
from vspy.features.get_formants_praat import get_formants_praat
from vspy.features.get_pitch_praat import get_pitch_praat
from vspy.features.get_pitch_snack import get_pitch_snack


# Analysis window length (ms) per feature, used for window-center time alignment.
# Snack extractors emit values indexed by frame START, so each value really
# belongs at frame_center = j + window/2 ms; the central shift below moves them
# there (the first window/2 ms become NaN), matching VoiceSauce. Praat extractors
# already report window-CENTER times via get_value_at_time / xs(), so they need
# no shift (WINDOW_MS = 0). Keeping this here means the alignment is defined once
# for the whole pipeline instead of being re-implemented inside each module.
WINDOW_MS = {
    "f0_praat": 0,
    "formants_praat": 0,
    "f0_snack": 25,
    "formants_snack": 25,
}

# define dictionary that maps feature name to extractor function
# uses string created by feature_key in api.py
REGISTRY = {
    "f0_praat": get_pitch_praat,
    "formants_praat": get_formants_praat,
    "f0_snack": get_pitch_snack,
    "formants_snack": get_formants_snack,
}


def _center_align(result, offset):
    """Shift frame-indexed output down by `offset` rows (prepend NaN, keep
    length), so frame-start-indexed values land at their window center. Handles
    both DataFrames (formants) and plain arrays (f0). offset == 0 is a no-op."""
    if offset <= 0:
        return result
    if isinstance(result, pd.DataFrame):
        out = result.copy()
        valcols = [c for c in out.columns if c != "t_ms"]
        out[valcols] = out[valcols].shift(offset)
        return out
    arr = np.asarray(result, dtype=float)
    shifted = np.full_like(arr, np.nan)
    if offset < len(arr):
        shifted[offset:] = arr[: len(arr) - offset]
    return shifted


# define a function that accepts feature names, wavfile, and shared args and calls each function
def run(features, wavfile, frameshift_ms, datalen):
    results = {}
    for name in features:
        extractor = REGISTRY[name]
        result = extractor(wavfile, frameshift_ms=frameshift_ms, datalen=datalen)
        # central window-center alignment (single source of truth for all features)
        offset = int(WINDOW_MS.get(name, 0) / frameshift_ms / 2)
        results[name] = _center_align(result, offset)
    #return a dictionary of arrays that api.py will use to build df
    return results
