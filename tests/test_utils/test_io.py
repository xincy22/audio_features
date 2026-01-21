import unittest
import tempfile
import os
import numpy as np
import soundfile as sf

from audiofeatures.utils import save_audio, save_features, load_features


class TestIOUtils(unittest.TestCase):
    def test_save_audio(self):
        sr = 8000
        t = np.linspace(0, 1, sr, endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            save_audio(signal, sr, path)
            loaded, loaded_sr = sf.read(path)
            self.assertEqual(loaded_sr, sr)
            self.assertEqual(len(loaded), len(signal))

    def test_save_and_load_features(self):
        features = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([[1.0, 2.0], [3.0, 4.0]])
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "features.npz")
            save_features(features, path)
            loaded = load_features(path)
            self.assertEqual(set(loaded.keys()), set(features.keys()))
            np.testing.assert_allclose(loaded["a"], features["a"])
            np.testing.assert_allclose(loaded["b"], features["b"])


if __name__ == "__main__":
    unittest.main()
