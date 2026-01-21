import unittest
import numpy as np

from audiofeatures.features import zero_crossing_rate, energy, log_energy, pitch


class TestTimeDomainFeatures(unittest.TestCase):
    def test_zero_crossing_rate(self):
        signal = np.array([1, -1, 1, -1, 1, -1], dtype=float)
        zcr = zero_crossing_rate(signal, frame_length=4, hop_length=2)
        self.assertEqual(zcr.ndim, 2)
        self.assertEqual(zcr.shape[1], 1)
        self.assertTrue(np.all(zcr >= 0))
        self.assertTrue(np.all(zcr <= 1))

    def test_energy_and_log_energy(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.array([1.0**2 + 2.0**2, 2.0**2 + 3.0**2, 3.0**2 + 4.0**2]).reshape(-1, 1)
        values = energy(signal, frame_length=2, hop_length=1)
        np.testing.assert_allclose(values, expected)
        log_values = log_energy(signal, frame_length=2, hop_length=1, eps=1e-10)
        np.testing.assert_allclose(log_values, np.log(expected + 1e-10))

    def test_pitch(self):
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)
        p = pitch(signal, sr=sr, frame_length=1024, hop_length=512)
        voiced = p[p > 0]
        self.assertTrue(voiced.size > 0)
        self.assertTrue(np.abs(np.median(voiced) - 440) < 30)


if __name__ == "__main__":
    unittest.main()
