import unittest
import numpy as np

from audiofeatures.features import (
    magnitude_spectrum,
    power_spectrum,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff
)


class TestFrequencyDomainFeatures(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.linspace(0, 1, self.sr, endpoint=False)
        self.signal = np.sin(2 * np.pi * 440 * t)

    def test_spectra_shapes(self):
        n_fft = 256
        hop_length = 128
        mag = magnitude_spectrum(self.signal, n_fft=n_fft, hop_length=hop_length)
        power = power_spectrum(self.signal, n_fft=n_fft, hop_length=hop_length)
        self.assertEqual(mag.shape[1], 1 + n_fft // 2)
        self.assertEqual(power.shape, mag.shape)
        np.testing.assert_allclose(power, mag ** 2)

    def test_spectral_features(self):
        centroid = spectral_centroid(self.signal, sr=self.sr, n_fft=1024, hop_length=512)
        bandwidth = spectral_bandwidth(self.signal, sr=self.sr, n_fft=1024, hop_length=512)
        rolloff = spectral_rolloff(self.signal, sr=self.sr, n_fft=1024, hop_length=512)
        self.assertEqual(centroid.ndim, 2)
        self.assertEqual(bandwidth.ndim, 2)
        self.assertEqual(rolloff.ndim, 2)
        self.assertEqual(centroid.shape[1], 1)
        self.assertEqual(centroid.shape, bandwidth.shape)
        self.assertEqual(centroid.shape, rolloff.shape)


if __name__ == "__main__":
    unittest.main()
