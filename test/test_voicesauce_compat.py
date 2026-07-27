"""Regression tests pinning vspy's CPP / HNR / Energy / SoE / SHR to VoiceSauce.

The golden-value tests drive the feature functions with the *same* F0 track
VoiceSauce used (strF0, read out of the reference output) so that they test the
measurement code rather than the F0 estimator, which is a separate substitution
(TANDEM-STRAIGHT is distributed as MATLAB P-code and cannot be ported).

Run with:  pytest test/test_voicesauce_compat.py
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from scipy.signal import lfilter, resample_poly

from vspy.api import SMOOTH_WIN, SMOOTHED_COLUMNS
from vspy.features._matlab import mround, mround_int, pickpeaks
from vspy.features.get_cpp import NPERIODS_EC, get_cpp
from vspy.features.get_energy import get_energy
from vspy.features.get_hnr import get_hnr
from vspy.features.get_shrp import get_shrp
from vspy.features.get_soe import get_soe
from vspy.io import read_wav

ROOT = Path(__file__).resolve().parent.parent
REF_CSV = ROOT / "test" / "output_vs.csv"
WAV_DIR = ROOT / "test" / "wav"

# A spread of speakers/tokens; enough to catch a systematic regression without
# making the suite slow.
GOLDEN_FILES = ["alice_b", "bean_b", "john_b", "satan_t_c", "curly_b"]


pytestmark = pytest.mark.skipif(
    not REF_CSV.exists(), reason="VoiceSauce reference output not available"
)


@pytest.fixture(scope="module")
def reference():
    return pd.read_csv(REF_CSV)


def _prepare(stem, reference):
    """Reproduce the VoiceSauce front end: resample to 16 kHz, write/re-read as
    16-bit PCM, and pull strF0 onto the frame grid."""
    y, fs = read_wav(WAV_DIR / f"{stem}.wav")
    g = math.gcd(int(fs), 16000)
    y16 = resample_poly(y, 16000 // g, int(fs) // g)

    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, y16, 16000, subtype="PCM_16")
        y16, _ = read_wav(Path(tmp))
    finally:
        os.unlink(tmp)

    datalen = int(len(y16) / 16000 * 1000)
    ref = reference[reference.Filename == f"{stem}.mat"]

    f0 = np.full(datalen, np.nan)
    idx = ref.t_ms.astype(int).to_numpy()
    keep = idx < datalen
    f0[idx[keep]] = ref.strF0.to_numpy()[keep]
    return y16, 16000, f0, datalen, ref


def _smooth(x):
    return lfilter([1.0 / SMOOTH_WIN] * SMOOTH_WIN, 1.0, np.asarray(x, dtype=float))


def _compare(values, datalen, ref, ref_col):
    merged = pd.merge(
        pd.DataFrame({"t_ms": np.arange(datalen), "got": values}),
        ref[["t_ms", ref_col]],
        on="t_ms",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    assert len(merged) > 100, f"too few comparable frames for {ref_col}"
    return merged["got"].to_numpy(), merged[ref_col].to_numpy()


def assert_reproduces(got, want, label):
    """Assert a VoiceSauce measure is reproduced frame for frame.

    The reference CSV prints strF0 with only three decimals, so on the rare
    frame where `round(ks +/- Nperiods/2*N0)` sits exactly on a .5 boundary the
    reconstructed F0 picks the other sample and the analysis window shifts by
    one sample.  The 20-frame output smoothing then spreads that single frame
    over 20 rows, so a couple of such frames per file shows up as ~10% of rows
    carrying a small (<0.1) error.  A real regression moves the median.
    """
    err = np.abs(np.asarray(got) - np.asarray(want))
    assert np.median(err) < 1e-3, f"{label}: median |error| = {np.median(err)}"
    assert err.mean() < 0.01, f"{label}: mean |error| = {err.mean()}"
    assert np.mean(err > 0.01) < 0.15, (
        f"{label}: {np.mean(err > 0.01):.1%} of frames off by > 0.01")
    assert err.max() < 0.2, f"{label}: max |error| = {err.max()}"


# ---------------------------------------------------------------------------
# Golden-value tests against the VoiceSauce reference output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", GOLDEN_FILES)
def test_cpp_matches_voicesauce(stem, reference):
    y, fs, f0, datalen, ref = _prepare(stem, reference)
    got, want = _compare(_smooth(get_cpp(y, fs, f0)), datalen, ref, "CPP")
    assert_reproduces(got, want, f"CPP/{stem}")


@pytest.mark.parametrize("stem", GOLDEN_FILES)
@pytest.mark.parametrize("band", [0, 1, 2, 3])
def test_hnr_matches_voicesauce(stem, band, reference):
    y, fs, f0, datalen, ref = _prepare(stem, reference)
    col = ["HNR05", "HNR15", "HNR25", "HNR35"][band]
    got, want = _compare(_smooth(get_hnr(y, fs, f0)[band]), datalen, ref, col)
    assert_reproduces(got, want, f"{col}/{stem}")


@pytest.mark.parametrize("stem", GOLDEN_FILES)
def test_energy_matches_voicesauce(stem, reference):
    y, fs, f0, datalen, ref = _prepare(stem, reference)
    # Energy is NOT in func_buildMData's smoothing list.
    got, want = _compare(get_energy(y, fs, f0), datalen, ref, "Energy")
    assert_reproduces(got, want, f"Energy/{stem}")


@pytest.mark.parametrize("stem", GOLDEN_FILES)
def test_soe_and_epoch_match_voicesauce(stem, reference):
    y, fs, f0, datalen, ref = _prepare(stem, reference)
    out = get_soe(y, fs, f0)

    merged = pd.merge(
        pd.DataFrame({"t_ms": np.arange(datalen),
                      "epoch_got": out["epoch"].to_numpy(),
                      "soe_got": out["soe"].to_numpy()}),
        ref[["t_ms", "epoch", "soe"]],
        on="t_ms",
    )
    # Epoch detection has to land on exactly the same frames.
    assert (merged.epoch_got == (merged.epoch > 0)).all()

    both = merged.dropna(subset=["soe_got", "soe"])
    assert len(both) > 20
    # SoE values carry a small residual from MATLAB resample() vs
    # scipy resample_poly() in the 16 kHz -> 10 kHz downsample.
    rel = np.abs(both.soe_got - both.soe) / np.abs(both.soe)
    assert np.median(rel) < 0.05
    assert np.corrcoef(both.soe_got, both.soe)[0, 1] > 0.999


@pytest.mark.parametrize("stem", GOLDEN_FILES)
def test_shr_matches_voicesauce(stem, reference):
    """SHR needs no external F0, so it should reproduce VoiceSauce exactly."""
    y, fs = read_wav(WAV_DIR / f"{stem}.wav")
    g = math.gcd(int(fs), 16000)
    y16 = resample_poly(y, 16000 // g, int(fs) // g)
    datalen = int(len(y16) / 16000 * 1000)

    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, y16, 16000, subtype="PCM_16")
        out = get_shrp(tmp, frameshift_ms=1, datalen=datalen)
    finally:
        os.unlink(tmp)

    ref = reference[reference.Filename == f"{stem}.mat"]
    got, want = _compare(out["shr"].to_numpy(), datalen, ref, "SHR")
    assert np.max(np.abs(got - want)) < 0.001


# ---------------------------------------------------------------------------
# Unit tests for the individual MATLAB behaviours that were being missed
# ---------------------------------------------------------------------------

def test_mround_is_half_away_from_zero():
    """Python/numpy round half to even; MATLAB rounds half away from zero."""
    assert mround_int(0.5) == 1
    assert mround_int(1.5) == 2
    assert mround_int(2.5) == 3      # Python's round(2.5) == 2
    assert mround_int(-0.5) == -1
    assert mround_int(-2.5) == -3
    np.testing.assert_array_equal(mround([0.5, 1.5, 2.5, 3.5]), [1, 2, 3, 4])


def test_pickpeaks_returns_one_based_positions():
    """func_pickpeaks.m returns 1-based indices; callers use them arithmetically."""
    y = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0])
    _, inx = pickpeaks(y, 1)  # winlen 1 -> no suppression between peaks
    # Peaks sit at 1-based 2, 4, 6, 8; the last one is dropped by inx1(end) = 0.
    np.testing.assert_array_equal(inx, [2, 4, 6])


def test_pickpeaks_rejects_edge_maxima():
    """inx1(1) = 0 and inx1(end) = 0 -- maxima at either end are not accepted."""
    y = np.array([5.0, 0.0, 1.0, 0.0, 5.0])
    _, inx = pickpeaks(y, 1)
    np.testing.assert_array_equal(inx, [3])


def test_pickpeaks_suppresses_smaller_peaks_within_window():
    y = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 9.0, 0.0])
    # Window covers everything, so only the largest accepted peak survives.
    # The 1-based-8 peak is rejected as an end maximum, leaving 2, 4, 6 -> 4.
    _, inx = pickpeaks(y, 100)
    np.testing.assert_array_equal(inx, [4])


def test_cpp_and_hnr_use_five_periods():
    """vs_Initialize.m: Nperiods_EC = 5 (not the Nperiods = 3 used elsewhere)."""
    from vspy.features import get_energy as energy_mod
    from vspy.features import get_hnr as hnr_mod

    assert NPERIODS_EC == 5
    assert hnr_mod.NPERIODS_EC == 5
    assert energy_mod.NPERIODS_EC == 5


def test_frame_zero_has_no_matlab_counterpart():
    """MATLAB's loop is k = 1..len with ks = round(k*sampleshift); vspy row i is
    t_ms = i, so row 0 (ks == 0) must stay NaN for every framewise measure."""
    fs = 16000
    rng = np.random.default_rng(0)
    y = rng.standard_normal(fs // 2) * 0.1
    f0 = np.full(200, 150.0)

    assert np.isnan(get_cpp(y, fs, f0)[0])
    assert np.isnan(get_energy(y, fs, f0)[0])
    assert all(np.isnan(band[0]) for band in get_hnr(y, fs, f0))


def test_frame_center_matches_matlab_ks():
    """Energy at row k must equal sum(y(ystart:yend).^2) for ks = k*sampleshift."""
    fs = 16000
    rng = np.random.default_rng(1)
    y = rng.standard_normal(fs) * 0.1
    f0_hz = 200.0
    f0 = np.full(400, f0_hz)
    energy = get_energy(y, fs, f0)

    k = 100
    ks = mround_int(k * fs / 1000.0)
    n0 = fs / f0_hz
    ystart = mround_int(ks - 5 / 2 * n0)
    yend = mround_int(ks + 5 / 2 * n0) - 1
    expected = np.sum(y[ystart - 1:yend] ** 2)
    assert energy[k] == pytest.approx(expected)


def test_shr_output_is_offset_by_two_frames(tmp_path):
    """func_GetSHRP.m computes n = round(k/frameshift)+1 and then writes to
    index n+1, so the value measured at t = k ms lands at t_ms = k+2.

    shrp.m anchors frame 0 at windowsize/2 = 12.5 ms, which rounds to t = 13;
    the first k that clears the frame_precision test is k = 12, giving
    n = 13 and an output row of 14.
    """
    fs = 16000
    t = np.arange(int(0.5 * fs)) / fs
    y = np.zeros_like(t)
    for h in range(1, 11):
        y += np.sin(2 * np.pi * 150 * h * t) / h
    y /= np.max(np.abs(y))
    wav = tmp_path / "tone.wav"
    sf.write(wav, y, fs, subtype="PCM_16")

    datalen = 500
    out = get_shrp(wav, frameshift_ms=1, datalen=datalen)
    shr = out["shr"].to_numpy()
    first = int(np.flatnonzero(np.isfinite(shr))[0])
    assert first == 14, f"first written row was {first}, expected 14"
    assert np.all(np.isnan(shr[:14]))


def test_shr_uses_voicesauce_f0_bounds():
    """vs_Initialize.m: SHRmin = 40, SHRmax = 500 (shrp.m's own defaults differ)."""
    import inspect

    sig = inspect.signature(get_shrp)
    assert sig.parameters["f0min"].default == 40.0
    assert sig.parameters["f0max"].default == 500.0
    assert sig.parameters["windowsize_ms"].default == 25.0
    assert sig.parameters["shr_threshold"].default == 0.4
    assert sig.parameters["ceiling"].default == 1250.0


def test_remove_trend_edge_windows_use_2n_minus_1_samples():
    """RemoveTrend's cbegin/cend both cumsum over width-2 == 2N-1 samples."""
    from vspy.features.get_soe import _remove_trend

    N = 3
    x = np.arange(1.0, 21.0)
    out = _remove_trend(x, N)
    trend = x - out
    # cbegin(1) = x(1)/1, cend(end) = x(L)/1
    assert trend[0] == pytest.approx(x[0])
    assert trend[-1] == pytest.approx(x[-1])
    # cbegin(2) = (x1+x2+x3)/3, cend(end-1) = (xL + xL-1 + xL-2)/3
    assert trend[1] == pytest.approx(x[:3].mean())
    assert trend[-2] == pytest.approx(x[-3:].mean())
    assert len(trend) == len(x)


def test_output_smoothing_covers_the_buildmdata_list():
    """func_buildMData.m smooths CPP/HNR and the harmonic measures, but leaves
    Energy, SHR, SoE, epoch, F0, formants and bandwidths alone."""
    assert SMOOTH_WIN == 20
    for col in ("CPP", "HNR05", "HNR15", "HNR25", "HNR35",
                "H1", "H2", "H4", "A1", "A2", "A3", "H2K", "H5K",
                "H1H2", "H2H4", "H1A1", "H1A2", "H1A3", "H4H2K", "H2KH5K"):
        assert col in SMOOTHED_COLUMNS
    for col in ("Energy", "shr", "SoE", "epoch", "f0_snack", "F1_snack", "B1_snack"):
        assert col not in SMOOTHED_COLUMNS
