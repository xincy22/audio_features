import unittest
import numpy as np

from audiofeatures.features import signal_statistics, spectral_statistics, power_spectrum


class TestStatisticalFeatures(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.linspace(0, 1, self.sr, endpoint=False)
        self.signal = np.sin(2 * np.pi * 440 * t)

    def test_signal_statistics(self):
        stats = signal_statistics(self.signal, frame_length=400, hop_length=160)
        self.assertIn("mean", stats)
        lengths = [len(value) for value in stats.values()]
        self.assertTrue(all(length == lengths[0] for length in lengths))

    def test_spectral_statistics(self):
        spec = power_spectrum(self.signal, n_fft=512, hop_length=256)
        stats = spectral_statistics(spec, sr=self.sr, n_fft=512)
        self.assertIn("centroid", stats)
        lengths = [len(value) for value in stats.values()]
        self.assertTrue(all(length == lengths[0] for length in lengths))


if __name__ == "__main__":
    unittest.main()
