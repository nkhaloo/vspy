import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from vspy import vspy
from vspy.features.get_spectral_moments import get_spectral_moments


class SpectralMomentsTests(unittest.TestCase):
    def test_tone_and_frame_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            fs = 48000
            t = np.arange(fs // 5) / fs
            sf.write(path, 0.5 * np.sin(2 * np.pi * 10000 * t), fs,
                     subtype="FLOAT")
            for hop in (1, 2.5):
                with patch("vspy.registry.REGISTRY", {
                    "spectral_moments": get_spectral_moments,
                    "f0_straight": lambda wavfile, **kw: np.zeros(kw["datalen"]),
                }):
                    df = vspy(path, frameshift_ms=hop,
                              output_file=Path(directory) / "out.csv")
                self.assertEqual(len(df), int(200 / hop))
                np.testing.assert_allclose(df.t_ms, np.arange(len(df)) * hop)
                self.assertTrue(np.isnan(df.spectral_cog.iloc[0]))
                self.assertTrue(np.isnan(df.spectral_cog.iloc[-1]))
                valid = df.spectral_cog.dropna()
                self.assertEqual(valid.index[0], int(np.ceil(12.5 / hop)))
                np.testing.assert_allclose(valid, 10000, atol=1)
                self.assertTrue((df.spectral_sdev.dropna() > 0).all())
                np.testing.assert_allclose(df.spectral_skew.dropna(), 0, atol=0.01)

    def test_power_weighting_and_excess_kurtosis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixture.wav"
            fs = 16000
            t = np.arange(fs // 5) / fs
            y = np.sin(2 * np.pi * 2000 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)
            sf.write(path, y / 2, fs, subtype="FLOAT")
            result = get_spectral_moments(path).iloc[100]
            # The two separated peaks have power weights 0.8 and 0.2.
            self.assertAlmostEqual(result.spectral_cog, 2400, delta=1)
            self.assertAlmostEqual(result.spectral_sdev, 800, delta=5)
            self.assertAlmostEqual(result.spectral_skew, 1.5, delta=0.02)
            self.assertAlmostEqual(result.spectral_kurt, 0.25, delta=0.03)

    def test_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            sf.write(path, np.zeros(1600), 16000)
            self.assertTrue(get_spectral_moments(path).isna().all().all())

    def test_window_expands_to_avoid_gaps_for_large_hop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            fs = 16000
            t = np.arange(fs) / fs
            sf.write(path, np.sin(2 * np.pi * 2000 * t), fs, subtype="FLOAT")

            result = get_spectral_moments(path, frameshift_ms=50)

            # A 50 ms effective window is centered on each 50 ms grid point.
            # Every interior grid point is therefore covered contiguously.
            self.assertTrue(result.iloc[1:-1].spectral_cog.notna().all())


if __name__ == "__main__":
    unittest.main()
