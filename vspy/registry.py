# registry for feature names and extraction functions
# used by api call to map features to their functions and call them
import numpy as np
import pandas as pd

from vspy.features.get_formants_snack import get_formants_snack
from vspy.features.get_formants_praat import get_formants_praat
from vspy.features.get_pitch_praat import get_pitch_praat
from vspy.features.get_pitch_snack import get_pitch_snack
from vspy.features.get_pitch_straight import get_pitch_straight
from vspy.features.get_shrp import get_shrp


# Per-feature output shift, in frames.
#
# Snack emits one value per analysis frame, indexed by frame START.  VoiceSauce
# realigns by prepending NaNs (vs_ParameterEstimation.m):
#
#     sF0 = [zeros(floor(windowsize/frameshift/2),1)*NaN; sF0];
#
# and then vs_OutputToText.m prints element `s` of that MATLAB (1-based) vector
# at t_ms = s.  So VoiceSauce's t_ms = i holds
#
#     snack_raw(i - floor(ws/fs/2))     -- 1-based
#   = snack frame  i - floor(ws/fs/2) - 1  -- 0-based
#
# vspy rows are 0-based with t_ms == row index, so the shift is
# floor(ws/fs/2) + 1 = 13 frames for a 25 ms window at a 1 ms hop.  The extra
# +1 is the 1-based/0-based fencepost, not a window-centre term: dropping it
# put every snack track one frame early.  Verified empirically -- with the +1
# our LPC poles land within 0.001 Hz of the reference sF1-sF4, without it they
# sit ~4 Hz off.
#
# Praat extractors report window-CENTRE times via get_value_at_time / xs(), and
# SHRP anchors at the window centre itself (its own +2 offset is applied inside
# get_shrp), so both need no shift here.
OFFSET_FRAMES = {
    "f0_praat": 0,
    "formants_praat": 0,
    "f0_snack": 13,
    "formants_snack": 13,
    "shrp": 0,
    # STRAIGHT reports each estimate at its own frame time (frame k <-> k ms),
    # like Praat, so no shift is needed.
    "f0_straight": 0,
}

# define dictionary that maps feature name to extractor function
# uses string created by feature_key in api.py
REGISTRY = {
    "f0_praat": get_pitch_praat,
    "formants_praat": get_formants_praat,
    "f0_snack": get_pitch_snack,
    "formants_snack": get_formants_snack,
    "shrp": get_shrp,
    "f0_straight": get_pitch_straight,
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
        # central time alignment (single source of truth for all features)
        results[name] = _center_align(result, OFFSET_FRAMES.get(name, 0))
    #return a dictionary of arrays that api.py will use to build df
    return results
