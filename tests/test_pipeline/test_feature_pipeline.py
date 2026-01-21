import unittest
import numpy as np

from audiofeatures.pipeline import FeatureExtractor, FeatureAggregator


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.linspace(0, 1, self.sr, endpoint=False)
        self.signal = np.sin(2 * np.pi * 440 * t)

    def test_feature_extractor(self):
        extractor = FeatureExtractor(sr=self.sr, n_fft=512, hop_length=256, n_mfcc=13)
        features = extractor.extract_features(self.signal, ["mfcc", "spectral_centroid", "zcr"])
        self.assertIn("mfcc", features)
        self.assertIn("spectral_centroid", features)
        self.assertIn("zcr", features)
        self.assertEqual(features["mfcc"].shape[0], 13)
        self.assertEqual(features["spectral_centroid"].ndim, 1)
        self.assertEqual(features["zcr"].ndim, 1)

    def test_feature_aggregator(self):
        features = {
            "mfcc": np.random.randn(13, 10),
            "zcr": np.random.rand(10)
        }
        aggregator = FeatureAggregator()
        aggregated = aggregator.aggregate_features(features, ["mean", "std"])
        self.assertIn("mfcc_mean", aggregated)
        self.assertIn("zcr_mean", aggregated)
        self.assertEqual(np.asarray(aggregated["mfcc_mean"]).shape, (13,))
        self.assertEqual(np.asarray(aggregated["zcr_mean"]).shape, ())

        stats = aggregator.aggregate_statistics(features)
        self.assertIn("mfcc", stats)
        self.assertIn("mean", stats["mfcc"])


if __name__ == "__main__":
    unittest.main()
