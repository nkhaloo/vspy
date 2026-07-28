"""Regression tests pinning get_pitch_snack to Snack's C source and to the
VoiceSauce reference output (sF0).

Sources of truth:
  snack2.2.10/generic/jkGetF0.c   -- Get_f0, check_f0_params, init_dp_f0, dp_f0,
                                     get_fast_cands, get_cand, peak,
                                     downsample/downsamp/do_ffir, lc_lin_fir,
                                     get_stationarity/get_similarity
  snack2.2.10/generic/sigproc.c   -- xhnwindow, xautoc, xdurbin, xa_to_aca,
                                     xitakura, wind_energy, xlpc, crossf, crossfi
  VoiceSauce/func_SnackPitch.m        -- the exact Tcl invocation
  VoiceSauce/vs_ParameterEstimation.m -- the NaN padding that sets the offset

Run with:  pytest test/test_snack_pitch_compat.py
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
from vspy.features import get_pitch_snack as G

ROOT = Path(__file__).resolve().parent.parent
REF_CSV = ROOT / "test" / "output_vs.csv"
WAV_DIR = ROOT / "test" / "wav"

GOLDEN_FILES = ["alice_b", "bean_b", "atlas_t_b", "bun_b", "alice_c"]

# The reference CSV prints sF0 with three decimals, so 5e-4 Hz is the finest
# agreement the file can express.
CSV_QUANTUM = 5.0e-4

pytestmark = pytest.mark.skipif(
    not REF_CSV.exists(), reason="VoiceSauce reference output not available"
)


@pytest.fixture(scope="module")
def reference():
    return pd.read_csv(REF_CSV, usecols=["Filename", "t_ms", "sF0"])


def _prepared(stem):
    """VoiceSauce front end: resample to 16 kHz and write 16-bit PCM.  Returns
    (temp wav path, datalen).  Caller unlinks."""
    y, fs = sf.read(WAV_DIR / f"{stem}.wav", dtype="float64", always_2d=True)
    g = math.gcd(int(fs), 16000)
    y16 = resample_poly(y[:, 0], 16000 // g, int(fs) // g)
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, y16, 16000, subtype="PCM_16")
    return tmp, int(len(y16) / 16000 * 1000)


def _run(stem):
    """get_pitch_snack on the registry's frame grid, aligned to VoiceSauce t_ms."""
    tmp, datalen = _prepared(stem)
    try:
        f0 = G.get_pitch_snack(Path(tmp), frameshift_ms=1, datalen=datalen)
    finally:
        os.unlink(tmp)
    f0 = registry._center_align(np.asarray(f0, float),
                                registry.OFFSET_FRAMES["f0_snack"])
    return pd.DataFrame({"t_ms": np.arange(datalen), "f0_snack": f0})


# ---------------------------------------------------------------------------
# Golden-value tests against the VoiceSauce reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", GOLDEN_FILES)
def test_f0_track_matches_voicesauce(stem, reference):
    """Assert on the error *distribution* over agreed-voiced frames.  Per-file
    R^2 is a bad threshold here because F0 barely varies within one file."""
    got = _run(stem)
    ref = reference[reference.Filename == f"{stem}.mat"]
    m = pd.merge(got, ref, on="t_ms")
    voiced = (m.f0_snack > 0) & (m.sF0 > 0)
    assert voiced.sum() > 100, f"{stem}: only {voiced.sum()} agreed-voiced frames"
    err = (m.f0_snack[voiced] - m.sF0[voiced]).abs()
    assert err.median() < CSV_QUANTUM, \
        f"{stem}: median |error| = {err.median():.6f} Hz"
    assert np.mean(err < 0.01) > 0.99, \
        f"{stem}: only {np.mean(err < 0.01):.2%} of frames within 0.01 Hz"


@pytest.mark.parametrize("stem", GOLDEN_FILES)
def test_voicing_decision_matches_voicesauce(stem, reference):
    """Voicing is inferred as sF0 > 0 (there is no sV column in the reference).
    Under-voicing was the dominant error mode before the front end was put on
    int16 and the stationarity LPC was fixed, so pin VDE hard."""
    got = _run(stem)
    ref = reference[reference.Filename == f"{stem}.mat"]
    m = pd.merge(got, ref, on="t_ms")
    ours = np.isfinite(m.f0_snack) & (m.f0_snack > 0)
    theirs = np.isfinite(m.sF0) & (m.sF0 > 0)
    vde = float(np.mean(ours != theirs))
    assert vde < 0.005, f"{stem}: VDE = {vde:.4f}"


@pytest.mark.parametrize("stem", GOLDEN_FILES)
def test_unvoiced_frames_are_zero_not_nan(stem, reference):
    """Snack writes 0 for an unvoiced frame (jkGetF0.c:1235); VoiceSauce keeps
    that and pads only the head/tail with NaN (vs_ParameterEstimation.m:687).
    Emitting NaN inside the tracked range would silently shrink n_finite in the
    evaluation and make the CSV differ from VoiceSauce's."""
    got = _run(stem)
    ref = reference[reference.Filename == f"{stem}.mat"]
    m = pd.merge(got, ref, on="t_ms")
    both = np.isfinite(m.f0_snack) & np.isfinite(m.sF0)
    assert both.sum() > 100
    # NaN must appear in exactly the same places as the reference's padding
    assert np.array_equal(np.isnan(m.f0_snack.to_numpy(float)),
                          np.isnan(m.sF0.to_numpy(float)))
    ours_zero = m.f0_snack[both] == 0
    theirs_zero = m.sF0[both] == 0
    assert (ours_zero == theirs_zero).all(), \
        f"{stem}: {(ours_zero != theirs_zero).sum()} frames disagree on zero"


# ---------------------------------------------------------------------------
# Stage-level test, independent of the DP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem", ["alice_b", "bean_b", "atlas_t_b"])
def test_reference_f0_is_among_the_generated_candidates(stem, reference):
    """Stronger than the track test and independent of the DP: the lag implied
    by the reference F0 must be present among the candidates `get_fast_cands`
    produced for that frame.  This is what catches a preprocessing / framing /
    NCCF regression -- e.g. taking lfilter(b,1,x)[::dec] instead of
    [ncoef//2::dec] mistimes the coarse pass by the decimator's 40-sample group
    delay and drops the true peak on a few percent of voiced frames."""
    tmp, _ = _prepared(stem)
    try:
        y, fs = G._read_int16(tmp)
    finally:
        os.unlink(tmp)

    freq = float(fs)
    step, size = 16, 400
    start, stop = G._eround(freq / 500), G._eround(freq / 40)
    nlags = stop - start + 1
    dec = G._ctrunc(freq / 2000.0)
    ds = G._downsample(y, freq, dec)

    ref = reference[reference.Filename == f"{stem}.mat"].set_index("t_ms")["sF0"]
    shift = registry.OFFSET_FRAMES["f0_snack"]
    n_frames = (len(y) - 841) // step

    dists = []
    for g in range(n_frames):
        t = g + shift
        if t not in ref.index:
            continue
        v = ref.loc[t]
        if not np.isfinite(v) or v <= 0:
            continue
        _, locs, _, _ = G._get_fast_cands(y, ds, g, step, size, dec, start,
                                          nlags, 20, 0.3, 0.3)
        if len(locs) == 0:
            dists.append(np.inf)
        else:
            dists.append(float(np.min(np.abs(locs - freq / v))))

    d = np.array(dists)
    assert len(d) > 150, f"{stem}: only {len(d)} reference-voiced frames"
    assert np.mean(d <= 1.0) > 0.95, (
        f"{stem}: reference lag present among candidates on only "
        f"{np.mean(d <= 1.0):.2%} of voiced frames")


def test_tracker_is_deterministic():
    """No RNG anywhere in the tracker; repeated runs must be bit-identical."""
    a = _run("bean_b")["f0_snack"].to_numpy(float)
    b = _run("bean_b")["f0_snack"].to_numpy(float)
    assert np.array_equal(a, b, equal_nan=True)


# ---------------------------------------------------------------------------
# Unit tests pinning each individual C behaviour
# ---------------------------------------------------------------------------

def test_voicesauce_overrides_only_four_snack_defaults():
    """func_SnackPitch.m emits
        s pitch -method esps -framelength 0.001 -windowlength 0.025 \
                -maxpitch 500 -minpitch 40
    which overrides min_f0/max_f0/wind_dur/frame_step (jkGetF0.c:82-85).  Every
    other parameter must keep its C default (jkGetF0.c:74-89)."""
    import inspect
    d = inspect.signature(G.get_pitch_snack).parameters
    assert d["min_f0"].default == 40           # -minpitch 40
    assert d["max_f0"].default == 500          # -maxpitch 500
    assert d["wind_dur"].default == 0.025      # -windowlength 0.025
    assert d["frameshift_ms"].default == 1     # -framelength 0.001
    assert d["cand_thresh"].default == 0.3     # jkGetF0.c:74
    assert d["lag_weight"].default == 0.3      # jkGetF0.c:75
    assert d["freq_weight"].default == 0.02    # jkGetF0.c:76
    assert d["trans_cost"].default == 0.005    # jkGetF0.c:77
    assert d["trans_amp"].default == 0.5       # jkGetF0.c:78
    assert d["trans_spec"].default == 0.5      # jkGetF0.c:79
    assert d["voice_bias"].default == 0.0      # jkGetF0.c:80
    assert d["double_cost"].default == 0.35    # jkGetF0.c:81
    assert d["n_cands"].default == 20          # jkGetF0.c:86


def test_eround_rounds_half_away_from_zero():
    """jkGetF0.c:1545-1550 `eround`.  Python/numpy round half to *even*, so
    round() is not a substitute."""
    assert G._eround(0.5) == 1 and round(0.5) == 0
    assert G._eround(2.5) == 3 and round(2.5) == 2
    assert G._eround(-0.5) == -1
    assert G._eround(-2.5) == -3
    # the derived analysis constants at 16 kHz / VoiceSauce settings
    assert G._eround(0.001 * 16000) == 16      # step
    assert G._eround(0.025 * 16000) == 400     # size
    assert G._eround(16000 / 500) == 32        # start
    assert G._eround(16000 / 40) == 400        # stop


def test_c_cast_truncates_toward_zero_in_coarse_lag_refinement():
    """jkGetF0.c:342 refines the coarse lag with

        *lp = (*lp * dec) + (int)(0.5 + (xp*dec));

    `(int)` truncates toward *zero*, so for a negative sub-sample offset this is
    emphatically not round(): xp*dec = -2.7 gives -2, while round() gives -3."""
    assert G._ctrunc(0.5 + -2.7) == -2
    assert round(0.5 + -2.7) == -2         # coincides here
    assert G._ctrunc(0.5 + -0.7) == 0
    assert int(np.round(-0.7)) == -1       # ... but not here
    assert G._ctrunc(0.5 + -1.3) == 0
    assert int(np.round(-1.3)) == -1
    assert G._ctrunc(0.5 + 2.7) == 3


def test_snack_hanning_has_the_half_sample_offset():
    """sigproc.c:165-166 `xhnwindow`: 0.5 - 0.5*cos((0.5+i)*2*pi/n).  np.hanning
    uses 2*pi*i/(n-1), whose endpoints are exactly zero."""
    n = 8
    w = G._snack_hanning(n)
    i = np.arange(n)
    assert np.allclose(w, 0.5 - 0.5 * np.cos((0.5 + i) * 2 * np.pi / n))
    assert w[0] > 0.0 and w[-1] > 0.0                  # never touches zero
    assert not np.allclose(w, np.hanning(n))
    assert np.allclose(w, w[::-1])                     # symmetric


def test_decimator_compensates_its_own_group_delay():
    """jkGetF0.c:506-613 `do_ffir` primes its delay line with ncoef-1 zeros and
    then ncoef samples, so output l is centred on input sample l*dec -- the
    81-tap filter's 40-sample group delay is removed.  A plain
    lfilter(b,1,x)[::dec] is centred on l*dec - 40 instead."""
    fs, dec = 16000.0, 8
    ncoeff = G._ctrunc(fs * 0.005) | 1
    assert ncoeff == 81
    b, nf = G._lc_lin_fir(0.5 / dec, ncoeff)
    assert nf == 81 and len(b) == 81
    assert np.allclose(b, b[::-1])                     # symmetric

    # An impulse at sample 800 must come back out centred at ds index 100.
    x = np.zeros(2000)
    x[800] = 10000.0
    ds = G._downsample(x, fs, dec)
    assert int(np.argmax(np.abs(ds - 0.5))) == 100


def test_decimator_keeps_the_leftover_integer_rounding_term():
    """jkGetF0.c:597 writes `(sum<0) ? sum-0.5 : sum+0.5` into a *float* buffer
    -- a rounding term left over from when the output held integers.  It
    survives into dsdata, so it must survive here."""
    x = np.zeros(2000)
    ds = G._downsample(x, 16000.0, 8)
    assert np.allclose(ds, 0.5)                        # silence -> +0.5, not 0


def test_fine_nccf_energy_floor_is_calibrated_to_int16():
    """sigproc.c:613 -- crossfi normalises by sqrt(10000 + engc*engref), not
    sqrt(engc*engref).

    The floor and the int16 front end are a matched pair, and this is the test
    that pins them together.  At int16 magnitudes 10000 is a genuine
    low-energy penalty: it is negligible for speech and suppresses frames near
    the quantisation floor.  Feed the *same waveform* scaled to [-1, 1] instead
    and the constant swamps engc*engref entirely, collapsing every fine NCCF to
    ~0 -- which is why an earlier port had to delete the floor to get any
    voicing at all, and why deleting it removed the only energy-dependent term
    from the fine correlation."""
    assert G.A_FACT == 10000.0
    size, start0, nlags0 = 400, 32, 369
    n = size + start0 + nlags0 + 32
    # Perfectly periodic at lag 100, so the un-floored NCCF is ~1.0 and the
    # only thing that can move it is the energy floor.
    wave = np.sin(2 * np.pi * np.arange(n) / 100.0)

    _, _, speech = G._crossfi(wave * 3000.0, size, start0, nlags0, 7, [100])
    _, _, silence = G._crossfi(wave * 1.0, size, start0, nlags0, 7, [100])
    _, _, floats = G._crossfi(wave / 32768.0, size, start0, nlags0, 7, [100])

    assert speech > 0.999                 # speech: floor is negligible
    assert silence < 0.9                  # ~1 LSB: measurably penalised
    assert floats < 1e-3                  # [-1,1] floats: annihilated

    # Without the floor all three would be ~1.0 -- silence would look exactly
    # as periodic as speech.
    saved = G.A_FACT
    try:
        G.A_FACT = 0.0
        _, _, unfloored = G._crossfi(wave * 1.0, size, start0, nlags0, 7, [100])
    finally:
        G.A_FACT = saved
    assert unfloored > 0.999


def test_crossfi_patch_is_always_seven_lags_wide():
    """sigproc.c:596-598 clamps the patch start at start0 but never clamps the
    end at stop, and always computes exactly `nlags` (7) lags."""
    size, start0, nlags0 = 40, 4, 30
    rng = np.random.default_rng(1)
    data = rng.standard_normal(size + start0 + nlags0 + 32) * 3000.0
    correl, _, _ = G._crossfi(data, size, start0, nlags0, 7, [5])
    # loc 5 - 3 = 2 < start0, so the patch starts at start0 and spans lags 4..10
    assert np.count_nonzero(correl[:7]) == 7
    assert np.all(correl[7:nlags0] == 0.0)


def test_get_cand_finds_shoulders_and_stops_two_short_of_nlags():
    """jkGetF0.c:445-486 `get_cand`: `q >= p && q >= o` accepts plateaus and
    shoulders as well as strict peaks, and the scan covers indices
    1 .. nlags-3 only (`lastl = nlags - 2`, with a three-deep primed window)."""
    n = 10
    c = np.zeros(n)
    c[3] = 1.0          # isolated peak
    c[4] = 1.0          # plateau partner -- also reported
    c[n - 2] = 5.0      # inside the array but past the scan window
    peaks, locs = G._get_cand(c, n, 100, 1.0, 0.3)
    assert list(locs) == [103, 104]
    assert list(peaks) == [1.0, 1.0]


def test_xlpc_normalises_the_autocorrelation():
    """sigproc.c:227,241 -- xautoc returns r[0] == 1 and r[i] = R_i/R_0, and
    `xitakura` depends on it: it starts its sum at c = 1 + sum(a^2) and adds
    only r[1..p] (sigproc.c:322-323), i.e. it multiplies the constant term by an
    implicit r[0] == 1.  So the C's Itakura is the quadratic form
    sum A[i]A[j] r[|i-j|] / gain *only* for a normalised r; hand it a raw
    autocorrelation and the distance is wrong by c*(r[0]-1)/gain."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500) * 2000.0
    a, rho, err = G._xlpc(18, 479, x)
    assert rho[0] == 1.0
    # Scaling the signal must leave rho and the residual untouched.
    _, rho2, err2 = G._xlpc(18, 479, x * 1000.0)
    assert np.allclose(rho, rho2)
    assert err2 == pytest.approx(err, rel=1e-9)

    # ... and the C's formulation breaks if r[0] != 1, while the equivalent
    # full double sum would not.  This is why the normalisation cannot be
    # dropped "because the ratio cancels" -- in xitakura it does not.
    A = np.concatenate([[1.0], a])
    good = G._itakura(a, rho, err, 18)
    scaled = 1e10
    bad = G._itakura(a, rho * scaled, err * scaled, 18)
    quad = sum(A[i] * A[j] * (rho * scaled)[abs(i - j)]
               for i in range(19) for j in range(19)) / (err * scaled)
    assert quad == pytest.approx(good, rel=1e-6)      # double sum: invariant
    assert abs(bad - good) > 1.0                      # xitakura: not invariant


def test_xlpc_stabilisation_factor():
    """sigproc.c:400-409 -- lpc_stabl = 30 (jkGetF0.c:1449) gives
    ffact = 1/(1 + 10**(-30/20)), applied to r[1..p] only, with r[0] left at 1.
    The *stabilised* r is written back through `ar`, so the Itakura distance and
    the residual are computed from the same, stabilised, autocorrelation."""
    assert G._FFACT == pytest.approx(1.0 / (1.0 + 10 ** (-30 / 20)))
    assert G._FFACT == pytest.approx(0.9693466, abs=1e-6)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(500) * 2000.0
    _, rho, _ = G._xlpc(18, 479, x)
    assert rho[0] == 1.0
    assert abs(rho[1]) < 1.0


def test_itakura_matches_the_quadratic_form():
    """sigproc.c:288-326 -- xa_to_aca + xitakura together compute
    sum_{i,j} A[i]A[j] rho[|i-j|] / gain for A = [1, a]."""
    rng = np.random.default_rng(4)
    p = 6
    a = rng.standard_normal(p) * 0.2
    rho = np.concatenate([[1.0], rng.standard_normal(p) * 0.3])
    A = np.concatenate([[1.0], a])
    want = sum(A[i] * A[j] * rho[abs(i - j)]
               for i in range(p + 1) for j in range(p + 1)) / 2.5
    assert G._itakura(a, rho, 2.5, p) == pytest.approx(want, rel=1e-12)


def test_itakura_is_not_clamped():
    """get_similarity (jkGetF0.c:1390,1404) computes t = itakura - 0.8 and
    returns 0.2/t with no floor.  An earlier port clamped itakura at 0.81, which
    caps the stationarity at 20 and makes every voicing transition cost ~10 --
    a pure under-voicing mechanism."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(1200) * 2000.0
    stats, ratios = G._get_stationarity(x, 16000.0, 40, 16)
    tail = stats[25:]
    assert len(tail) > 0
    assert np.all(tail < 20.0), f"stationarity pinned at the clamp: {tail.max()}"


def test_stationarity_leading_frame_special_cases():
    """jkGetF0.c:1504-1514 -- in the first buffer, frames whose *previous*
    30 ms window starts before the data get init=1 (rms_ratio 1.0, t = 10, so
    stat = 0.02), and frames whose *current* window also starts before the data
    get stat = 0.01*0.2 and rms_ratio = 1.0.  With ind = -80 and step = 16 those
    boundaries fall at frames 5 and 25."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal(4000) * 2000.0
    stats, ratios = G._get_stationarity(x, 16000.0, 60, 16)
    assert np.all(stats[:5] == pytest.approx(0.002))
    assert np.all(ratios[:5] == 1.0)
    assert np.all(stats[5:25] == pytest.approx(0.02))
    assert np.all(ratios[5:25] == 1.0)
    assert not np.allclose(stats[25:], 0.02)


def test_stationarity_window_geometry():
    """jkGetF0.c:1453-1455 -- stat_wsize = 480, agap = 320 and
    ind = (agap - stat_wsize)/2 = -80 at 16 kHz, so both 30 ms windows start
    *before* the frame and their centres are 20 ms apart."""
    fs = 16000.0
    stat_wsize = G._ctrunc(0.030 * fs)
    agap = G._ctrunc(0.020 * fs)
    assert (stat_wsize, agap) == (480, 320)
    assert int(agap - stat_wsize) // 2 == -80
    assert agap // 16 == 20                 # wReuse: windstat depth, in frames
    assert G._ctrunc(2.0 + fs / 1000.0) == 18       # LPC order


def test_frame_count_matches_the_read_loop():
    """jkGetF0.c:847-855 + :763-771 -- pad = downpatch + max(stat_wsize+ind,
    ncomp) = 40 + 801 = 841 at 16 kHz, and get_Nframes returns
    (buff_size - pad)/step per buffer.  Telescoped over Get_f0's read loop that
    is (N - pad)/step whole frames -- two fewer than a naive
    (N - size - stop)/step, which is where 300 spuriously voiced tail frames
    used to come from."""
    fs = 16000.0
    size, stop = 400, 400
    ncomp = size + stop + 1
    downpatch = (G._ctrunc(fs * 0.005) + 1) // 2
    stat_wsize, agap = 480, 320
    ind = int(agap - stat_wsize) // 2
    pad = downpatch + max(stat_wsize + ind, ncomp)
    assert (downpatch, ncomp, pad) == (40, 801, 841)

    rng = np.random.default_rng(7)
    n = 40000
    sig = (rng.standard_normal(n) * 2000).astype(np.int16)
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, sig, 16000, subtype="PCM_16")
        f0 = G.get_pitch_snack(Path(tmp), frameshift_ms=1, datalen=n // 16)
    finally:
        os.unlink(tmp)
    assert int(np.sum(~np.isnan(f0))) == (n - pad) // 16


def test_frame_step_is_snapped_to_whole_samples():
    """jkGetF0.c:292 -- check_f0_params replaces frame_step with
    ((int)(0.5 + sf*frame_step))/sf before anything else uses it."""
    fs = 16000.0
    assert G._ctrunc(0.5 + fs * 0.001) / fs == 0.001
    assert G._ctrunc(0.5 + fs * 0.0001) / fs == pytest.approx(2 / fs)


def test_front_end_reads_int16_sample_values():
    """Snack hands `Snack_GetSoundData` the raw LIN16 shorts
    (jkSoundEdit.c:232), so every energy in the tracker -- and the 10000 floor
    in crossfi -- is calibrated to int16 magnitudes.  soundfile scales a float
    read by 1/32768, so recover the integers with 32768, not 32767."""
    vals = np.array([0, 1, -1, 32767, -32768, 1234], dtype=np.int16)
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, vals, 16000, subtype="PCM_16")
        y, fs = G._read_int16(tmp)
    finally:
        os.unlink(tmp)
    assert fs == 16000
    assert np.array_equal(y, vals.astype(np.float64))
    assert np.abs(y).max() > 1.0


def test_dp_weights_are_normalised_by_stop_and_frame_interval():
    """jkGetF0.c:817,820,322 -- three different normalisers, easy to conflate:
    the DP lag weight divides by `stop`, the frequency weight by the frame
    interval, and get_fast_cands' own coarse lag weight by `nlags`."""
    start, stop = G._eround(16000 / 500), G._eround(16000 / 40)
    nlags = stop - start + 1
    assert (start, stop, nlags) == (32, 400, 369)
    assert 0.3 / stop == pytest.approx(0.00075)        # lagwt
    assert 0.02 / 0.001 == pytest.approx(20.0)         # freqwt
    assert 0.3 / nlags == pytest.approx(0.3 / 369)     # coarse lag_wt


def test_snack_offset_is_window_half_plus_one():
    """vs_ParameterEstimation.m:687 prepends floor(windowsize/frameshift/2)
    NaNs to a 1-based MATLAB vector and vs_OutputToText.m prints element s at
    t_ms = s; vspy rows are 0-based with t_ms == row, so the shift is + 1."""
    assert registry.OFFSET_FRAMES["f0_snack"] == 13
    assert math.floor(25 / 1 / 2) + 1 == 13
