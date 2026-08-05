

import numpy as np
import pandas as pd

from vspy.features.get_formants_snack import get_formants_snack
from vspy.features.get_formants_praat import get_formants_praat
from vspy.features.get_pitch_praat import get_pitch_praat
from vspy.features.get_pitch_snack import get_pitch_snack
from vspy.features.get_shrp import get_shrp

























OFFSET_FRAMES = {
    "f0_praat": 0,
    "formants_praat": 0,
    "f0_snack": 13,
    "formants_snack": 13,
    "shrp": 0,
}



REGISTRY = {
    "f0_praat": get_pitch_praat,
    "formants_praat": get_formants_praat,
    "f0_snack": get_pitch_snack,
    "formants_snack": get_formants_snack,
    "shrp": get_shrp,
}


def _center_align(result, offset):
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



def run(features, wavfile, frameshift_ms, datalen):
    results = {}
    for name in features:
        extractor = REGISTRY[name]
        result = extractor(wavfile, frameshift_ms=frameshift_ms, datalen=datalen)

        results[name] = _center_align(result, OFFSET_FRAMES.get(name, 0))

    return results
