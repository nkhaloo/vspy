import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vspy import registry
from vspy.features.get_nasality import NASALITY_COLUMNS, get_nasality


def test_nasality_is_part_of_default_feature_registry():
    assert "nasality" in registry.REGISTRY


def test_nasality_uses_25_ms_windows_by_default():
    parameter = inspect.signature(get_nasality).parameters["spectrum_window_s"]
    assert parameter.default == 0.025


def test_registry_routes_nasality_options_and_original_audio(monkeypatch):
    seen = {}

    def fake_nasality(source, frameshift_ms, datalen, **options):
        seen.update(
            source=source,
            frameshift_ms=frameshift_ms,
            datalen=datalen,
            options=options,
        )
        return pd.DataFrame(index=range(datalen), columns=NASALITY_COLUMNS)

    monkeypatch.setitem(registry.REGISTRY, "nasality", fake_nasality)
    original = Path("original.wav")
    result = registry.run(
        ["nasality"],
        Path("resampled.wav"),
        frameshift_ms=2,
        datalen=12,
        nasality_wavfile=original,
        nasality_options={"p1search": 900, "tpnum": 3, "scoot": 0.01},
    )

    assert seen == {
        "source": original,
        "frameshift_ms": 2,
        "datalen": 12,
        "options": {"p1search": 900, "tpnum": 3, "scoot": 0.01},
    }
    assert list(result["nasality"].columns) == list(NASALITY_COLUMNS)


def test_nasality_keeps_end_measurement_on_main_frame_grid(tmp_path):
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    wav = tmp_path / "vowel.wav"
    sf.write(wav, np.sin(2 * np.pi * 150 * time), sample_rate)

    result = get_nasality(wav, datalen=1000, tpnum=5)

    assert result["vwl_duration"].notna().sum() == 5
    assert result.loc[999, "vwl_duration"] == 1000


def test_nasality_supports_hops_larger_than_base_window(tmp_path):
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    wav = tmp_path / "vowel.wav"
    sf.write(wav, np.sin(2 * np.pi * 150 * time), sample_rate)

    result = get_nasality(wav, frameshift_ms=50, datalen=20, tpnum=5)

    assert len(result) == 20
    assert result["vwl_duration"].notna().sum() == 5
