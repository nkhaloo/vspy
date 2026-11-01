

import numpy as np
import pandas as pd

from vspy.features.get_formants_snack import get_formants_snack
from vspy.features.get_formants_praat import get_formants_praat
from vspy.features.get_pitch_praat import get_pitch_praat
from vspy.features.get_pitch_snack import get_pitch_snack
from vspy.features.get_pitch_straight import get_pitch_straight
from vspy.features.get_shrp import get_shrp
from vspy.features.get_spectral_moments import get_spectral_moments
from vspy.features.get_nasality import get_nasality

























OFFSET_FRAMES = {
    # VoiceSauce exports MATLAB element 1 at t=1 ms. Downstream vspy
    # windows use their frame index as time, so retain an empty frame zero.
    "f0_straight": 1,
    "f0_praat": 0,
    "formants_praat": 0,
    "f0_snack": 13,
    "formants_snack": 13,
    "shrp": 0,
}



REGISTRY = {
    "f0_straight": get_pitch_straight,
    "f0_praat": get_pitch_praat,
    "formants_praat": get_formants_praat,
    "f0_snack": get_pitch_snack,
    "formants_snack": get_formants_snack,
    "shrp": get_shrp,
    "spectral_moments": get_spectral_moments,
    "nasality": get_nasality,
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



def run(features, wavfile, frameshift_ms, datalen, straight_options=None,
        spectral_wavfile=None, nasality_options=None, nasality_wavfile=None):
    results = {}
    for name in features:
        extractor = REGISTRY[name]
        if name == "f0_straight":
            options = straight_options or {}
        elif name == "nasality":
            options = nasality_options or {}
        else:
            options = {}
        if name == "spectral_moments" and spectral_wavfile is not None:
            source = spectral_wavfile
        elif name == "nasality" and nasality_wavfile is not None:
            source = nasality_wavfile
        else:
            source = wavfile
        result = extractor(source, frameshift_ms=frameshift_ms, datalen=datalen, **options)

        results[name] = _center_align(result, OFFSET_FRAMES.get(name, 0))

    return results
