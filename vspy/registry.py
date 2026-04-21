# registry for feature names and extraction functions 
# used by api call to map features to their functions and cal them
from vspy.features.get_formants_snack import get_formants_snack
from vspy.features.get_formants_praat import get_formants_praat
from vspy.features.get_pitch_praat import get_pitch_praat
from vspy.features.get_pitch_pyin import get_pitch_pyin
from vspy.features.get_pitch_snack import get_pitch_snack


# define dictionary that maps feature name to extractor function
# uses string created by feature_key in api.py
REGISTRY = {
    "f0_praat": get_pitch_praat,
    #"f0_pyin": get_pitch_pyin,
    "formants_praat": get_formants_praat,
    "f0_snack": get_pitch_snack, 
    "formants_snack": get_formants_snack
}

# define a function that accepts feature names, wavfile, and shared args and calls each function
def run(features, wavfile, frameshift_ms, datalen):
    results = {}
    for name in features:
        extractor = REGISTRY[name]
        results[name] = extractor(wavfile, frameshift_ms=frameshift_ms, datalen=datalen)
    #return a dictionary of arrays that api.py will use to build df
    return results