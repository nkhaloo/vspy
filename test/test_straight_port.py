"""STRAIGHT reference accuracy and F0 routing regression tests."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vspy.api import _resample
from vspy.features.get_pitch_straight import get_pitch_straight, straight_f0
from vspy.io import read_wav
from vspy.align.textgrid import parse_textgrid


@pytest.mark.parametrize('stem', ['alice_b', 'bean_b', 'john_b', 'satan_t_c',
                                 'faint_t_c', 'greatness_t_c', 'bun_c', 'button_t_c'])
def test_voicesauce_export(stem, tmp_path):
    reference_path = Path('test/output_vs.csv')
    source = Path('test/wav') / (stem + '.wav')
    if not reference_path.exists() or not source.exists():
        pytest.skip('Restored VoiceSauce corpus is not installed')
    y, fs = read_wav(source)
    y, fs = _resample(y, fs)
    wav = tmp_path / 'input.wav'
    sf.write(wav, y, fs, subtype='PCM_16')
    intervals = parse_textgrid(Path('test/textgrid') / (stem + '.TextGrid'), 'words', round_times=False)
    actual = get_pitch_straight(wav, intervals=intervals, ignore_labels=('', ' ', 'SIL', '[SIL]'))
    ref = pd.read_csv(reference_path)
    ref = ref[ref.Filename == stem + '.mat']
    a = actual[ref.t_ms.to_numpy(int) - 1]
    b = ref.strF0.to_numpy()
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    # VoiceSauce exports three decimals: half a last-place unit is expected.
    np.testing.assert_allclose(a, b, atol=0.00051, rtol=0, equal_nan=True)


def test_periodic_signal_and_silence():
    fs = 16000
    t = np.arange(fs // 4) / fs
    x = sum(np.sin(2*np.pi*200*k*t) / k for k in range(1, 8))
    f0 = straight_f0(x, fs)
    np.testing.assert_allclose(f0[40:-40], 200, atol=0.15, rtol=0)
    assert np.isnan(straight_f0(np.zeros(100), fs)).all()
    assert len(straight_f0(np.empty(0), fs)) == 0


def test_straight_drives_every_dependent_feature(monkeypatch, tmp_path):
    api = importlib.import_module('vspy.api')
    length = 100
    straight = np.linspace(150, 170, length)
    other = np.full(length, 300.0)
    wav = tmp_path / 'routing.wav'
    sf.write(wav, np.zeros(1600), 16000)
    calls = []

    def extract(features, *args, **kwargs):
        assert 'f0_straight' in features
        return {'f0_straight': straight, 'f0_praat': other,
                'formants_praat': pd.DataFrame({f'F{i}_praat': np.full(length, i*500.) for i in (1, 2, 3)})}

    monkeypatch.setattr(api.registry, 'run', extract)

    def check(name, count):
        def function(y, fs, f0, *args, **kwargs):
            np.testing.assert_array_equal(f0, straight)
            calls.append(name)
            return tuple(np.zeros(length) for _ in range(count)) if count > 1 else np.zeros(length)
        return function

    for name, count in [('get_harmonics', 3), ('get_formant_amplitudes', 3),
                        ('get_2k5k', 3), ('get_cpp', 1), ('get_energy', 1), ('get_hnr', 4)]:
        monkeypatch.setattr(api, name, check(name, count))

    def tilts(*args):
        np.testing.assert_array_equal(args[9], straight)
        calls.append('compute_tilts')
        return {}

    def soe(y, fs, f0, **kwargs):
        np.testing.assert_array_equal(f0, straight)
        calls.append('get_soe')
        return pd.DataFrame({'epoch': np.zeros(length), 'soe': np.zeros(length)})

    monkeypatch.setattr(api, 'compute_tilts', tilts)
    monkeypatch.setattr(api, 'get_soe', soe)
    api.vspy(wav, features=['formants_praat'], output_csv=tmp_path / 'out.csv')
    assert set(calls) == {'get_harmonics', 'get_formant_amplitudes', 'get_2k5k',
                          'get_cpp', 'get_energy', 'get_hnr', 'compute_tilts', 'get_soe'}


def test_segment_placement_and_frameshift(monkeypatch, tmp_path):
    module = importlib.import_module('vspy.features.get_pitch_straight')
    wav = tmp_path / 'segments.wav'
    sf.write(wav, np.zeros(1600), 16000)
    seen = []

    def calculate(y, fs, *args):
        seen.append(len(y))
        return np.arange(int((len(y)-1) / fs * 1000) + 1) + 100.

    monkeypatch.setattr(module, 'straight_f0', calculate)
    f0 = get_pitch_straight(wav, intervals=[(30.5, 60.2, 'word'), (70, 80, 'SIL')])
    assert seen == [1277]  # MATLAB y(floor(5.5ms*fs):ceil(85.2ms*fs)).
    assert np.isnan(f0[:4]).all()
    assert f0[4] == 100
    sparse = get_pitch_straight(wav, frameshift_ms=5)
    np.testing.assert_array_equal(sparse, np.arange(0, 100, 5) + 100)
