"""Regression tests pinning get_formants_snack to Snack's C source and to the
VoiceSauce reference output (sF1-sF4 / sB1-sB4).

Sources of truth:
  snack2.2.10/generic/jkFormant.c   -- Fdownsample, highpass, lpc_poles,
                                       lpcbsa, get_fcand/candy, dpform
  snack2.2.10/generic/sigproc2.c    -- formant(), dcovlpc, dcwmtrx, dreflpc
  VoiceSauce/vs_ParameterEstimation.m -- the NaN padding that sets the offset

Run with:  pytest test/test_snack_formants_compat.py
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from vspy import registry
from vspy.features import get_formants_snack as G
from vspy.io import read_wav

ROOT = Path(__file__).resolve().parent.parent
REF_CSV = ROOT / "test" / "output_vs.csv"
WAV_DIR = ROOT / "test" / "wav"

GOLDEN_FILES = ["alice_b", "bean_b", "john_b", "atlas_t_b", "curly_b"]

pytestmark = pytest.mark.skipif(
    not REF_CSV.exists(), reason="VoiceSauce reference output not available"
)


@pytest.fixture(scope="module")
def reference():
    return pd.read_csv(REF_CSV)


def _run(stem):
    """VoiceSauce front end (resample to 16 kHz, write 16-bit) then snack
    formants on the registry's frame grid."""
    y, fs = read_wav(WAV_DIR / f"{stem}.wav")
    g = math.gcd(int(fs), 16000)
    y16 = resample_poly(y, 16000 // g, int(fs) // g)
    datalen = int(len(y16) / 16000 * 1000)
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, y16, 16000, subtype="PCM_16")
        df = G.get_formants_snack(Path(tmp), frameshift_ms=1, datalen=datalen)
    finally:
        os.unlink(tmp)
    df = registry._center_align(df, registry.OFFSET_FRAMES["formants_snack"])
    df["t_ms"] = np.arange(datalen)
    return df


# ---------------------------------------------------------------------------
# Golden-value tests against the VoiceSauce reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", GOLDEN_FILES)
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_formant_track_matches_voicesauce(stem, k, reference):
    got = _run(stem)
    ref = reference[reference.Filename == f"{stem}.mat"]
    m = pd.merge(got, ref, on="t_ms").dropna(subset=[f"F{k}_snack", f"sF{k}"])
    assert len(m) > 100
    err = (m[f"F{k}_snack"] - m[f"sF{k}"]).abs()
    # The tracker is exact frame-for-frame on the overwhelming majority of
    # frames (median error ~0.002 Hz); a small tail of DP path divergence
    # survives in hard regions. Per-file correlation is a poor threshold here
    # because within one file the tracks barely vary, so assert on the error
    # distribution instead and leave r2 to the corpus-level evaluation.
    assert err.median() < 0.05, f"F{k}/{stem}: median |error| = {err.median():.4f} Hz"
    assert np.mean(err < 5.0) > 0.90, (
        f"F{k}/{stem}: only {np.mean(err < 5.0):.1%} of frames within 5 Hz")


@pytest.mark.parametrize("stem", GOLDEN_FILES)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_bandwidth_track_matches_voicesauce(stem, k, reference):
    got = _run(stem)
    ref = reference[reference.Filename == f"{stem}.mat"]
    m = pd.merge(got, ref, on="t_ms").dropna(subset=[f"B{k}_snack", f"sB{k}"])
    assert len(m) > 100
    err = (m[f"B{k}_snack"] - m[f"sB{k}"]).abs()
    assert err.median() < 0.05, f"B{k}/{stem}: median |error| = {err.median():.4f} Hz"
    assert np.mean(err < 5.0) > 0.90, (
        f"B{k}/{stem}: only {np.mean(err < 5.0):.1%} of frames within 5 Hz")


@pytest.mark.parametrize("stem", ["alice_b", "bean_b", "atlas_t_b"])
def test_poles_reproduce_reference_formants(stem, reference):
    """Stronger than the track test: every reference formant must coincide with
    one of our LPC poles, independent of how the DP assigns them. This is what
    catches a regression in preprocessing / framing / the LPC solve."""
    y, fs = read_wav(WAV_DIR / f"{stem}.wav")
    g = math.gcd(int(fs), 16000)
    y16 = resample_poly(y, 16000 // g, int(fs) // g)
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, y16, 16000, subtype="PCM_16")
        sig, fs2 = sf.read(tmp, dtype="int16", always_2d=True)
    finally:
        os.unlink(tmp)
    proc, rate = G.preprocess(sig[:, 0], fs2, 10000.0)
    poles = G.lpc_poles(proc, rate, 0.001, 12)

    ref = reference[reference.Filename == f"{stem}.mat"].set_index("t_ms")
    shift = registry.OFFSET_FRAMES["formants_snack"]
    dists = []
    for j, p in enumerate(poles):
        t = j + shift
        if t not in ref.index or not p["npoles"]:
            continue
        for k in range(4):
            want = ref.loc[t, f"sF{k+1}"]
            if np.isfinite(want):
                dists.append(np.min(np.abs(p["freq"] - want)))
    d = np.array(dists)
    assert len(d) > 400
    assert np.median(d) < 0.05, f"{stem}: median pole distance = {np.median(d)} Hz"
    assert np.mean(d < 1.0) > 0.95, f"{stem}: only {np.mean(d<1.0):.1%} within 1 Hz"


# ---------------------------------------------------------------------------
# Unit tests for the individual Snack behaviours that were being missed
# ---------------------------------------------------------------------------

def test_dp_weights_use_the_frame_rate_not_the_audio_rate(monkeypatch):
    """jkFormant.c lpc_poles() returns the poles in a Sound created with
    Snack_NewSound((int)(1.0/frame_int), ...), so dpform() sees samprate =
    frame rate (1000 at a 1 ms hop). Passing the 10 kHz audio rate makes
    bfact/ffact 10x too small and dffact 10x too large."""
    seen = {}

    def fake_dpform(poles, nform, samprate):
        seen["samprate"] = samprate
        return np.zeros((nform, len(poles))), np.zeros((nform, len(poles)))

    monkeypatch.setattr(G, "dpform", fake_dpform)
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(16000) * 3000).astype(np.int16)
    G.get_formants(sig, 16000, 0.001, 12, 4, 10000.0)
    assert seen["samprate"] == 1000

    G.get_formants(sig, 16000, 0.01, 12, 4, 10000.0)
    assert seen["samprate"] == 100


def test_dp_weight_values():
    """Spot-check the three derived weights at Snack's real frame rate."""
    samprate = 1000.0
    assert G.DF_FACT * 0.01 * samprate == pytest.approx(200.0)
    assert G.BAND_FACT / (0.01 * samprate) == pytest.approx(2e-4)
    assert G.DFN_FACT / (0.01 * samprate) == pytest.approx(0.03)


def test_lpcbsa_gets_the_full_snack_buffer(monkeypatch):
    """lpcbsa() in the C reads `wind + np + 1` samples straight out of the
    signal buffer, i.e. lpc_ord+1 samples past the nominal window end."""
    lengths = []
    real = G.lpcbsa

    def spy(lpc_ord, wind, data, preemp, rng):
        lengths.append(len(data))
        return real(lpc_ord, wind, data, preemp, rng)

    monkeypatch.setattr(G, "lpcbsa", spy)
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(10000) * 3000).astype(np.int16)
    G.lpc_poles(sig, 10000, 0.001, 12)
    # 25 ms at 10 kHz = 250 samples, plus lpc_ord + 1 = 13
    assert max(lengths) == 250 + 12 + 1


def test_snack_offset_is_window_half_plus_one():
    """vs_ParameterEstimation.m prepends floor(windowsize/frameshift/2) NaNs to
    a 1-based MATLAB vector, and vs_OutputToText.m prints element s at t_ms=s.
    vspy rows are 0-based with t_ms == row, so the shift is that + 1."""
    assert registry.OFFSET_FRAMES["formants_snack"] == 13
    assert registry.OFFSET_FRAMES["f0_snack"] == 13
    assert registry.OFFSET_FRAMES["f0_praat"] == 0
    assert registry.OFFSET_FRAMES["formants_praat"] == 0
    assert registry.OFFSET_FRAMES["shrp"] == 0
    assert registry.OFFSET_FRAMES["f0_straight"] == 0


def test_tracker_is_deterministic():
    """Snack never calls srand(), so rand() replays the same stream in every
    fresh `wish` process. Repeated runs must give identical output."""
    a = _run("bean_b")
    b = _run("bean_b")
    for c in a.columns:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        assert np.array_equal(x[m], y[m]), f"{c} is not reproducible"


def test_preprocess_recovers_exact_int16_samples():
    """Snack works on the stored LIN16 integers. int16 input must pass through
    the scaling branch untouched, and float input must scale by 32768."""
    sig = np.array([-32768, -20000, -1, 0, 1, 20000, 32767], dtype=np.int16)
    out_i, _ = G.preprocess(sig, 10000.0, 10000.0)   # ds_freq == samprate: no resample
    out_f, _ = G.preprocess(sig.astype(np.float64) / 32768.0, 10000.0, 10000.0)
    np.testing.assert_array_equal(out_i, out_f)


def test_snack_uses_a_seven_digit_pi_in_its_fir_designs():
    """downsample.c / highpass() use `#define PI 3.1415927`, not M_PI."""
    assert G.SNACK_PI == 3.1415927
    assert G.SNACK_PI != math.pi


def test_domerge_is_disabled_by_the_default_merge_cost():
    """dpform(): `if(merge_cost > 1000.0) domerge = FALSE;` with F_MERGE = 2000,
    which also switches off the F1/F2 merge branch inside candy()."""
    assert G.F_MERGE == 2000.0
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(6000) * 3000).astype(np.int16)
    poles = G.lpc_poles(sig, 10000, 0.001, 12)
    G.dpform(poles, 4, 1000)
    assert G.domerge is False


def test_formant_returns_only_in_band_complex_poles():
    """sigproc2.c formant(): *n_form counts poles with 1 < f < Nyquist-1, and
    dpform only ever looks at the first *n_form entries."""
    rng = np.random.default_rng(1)
    sig = (rng.standard_normal(10000) * 3000).astype(np.int16)
    poles = G.lpc_poles(sig, 10000, 0.001, 12)
    for p in poles:
        f = p["freq"]
        assert len(f) == p["npoles"]
        if len(f):
            assert np.all(f > 1.0) and np.all(f < 5000.0 - 1.0)
            assert np.all(np.diff(f) >= 0)      # sorted ascending
