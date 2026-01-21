import unittest
import numpy as np

from audiofeatures.augmentation import add_noise, time_mask


class TestAugmentationTimeDomain(unittest.TestCase):
    def test_add_noise_deterministic_with_seed(self):
        signal = np.zeros(64, dtype=np.float32)
        out1 = add_noise(signal, noise_level=0.01, seed=123)
        out2 = add_noise(signal, noise_level=0.01, seed=123)
        np.testing.assert_allclose(out1, out2)

    def test_time_mask_deterministic_with_seed(self):
        signal = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
        out1 = time_mask(signal, mask_fraction=0.2, seed=7)
        out2 = time_mask(signal, mask_fraction=0.2, seed=7)
        np.testing.assert_allclose(out1, out2)

    def test_rng_and_seed_are_mutually_exclusive(self):
        signal = np.zeros(16, dtype=np.float32)
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            add_noise(signal, rng=rng, seed=0)
        with self.assertRaises(ValueError):
            time_mask(signal, rng=rng, seed=0)


if __name__ == "__main__":
    unittest.main()
