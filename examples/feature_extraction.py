"""
Feature extraction example.
"""

import numpy as np

from audiofeatures.utils import load_audio
from audiofeatures.pipeline import FeatureExtractor, FeatureAggregator

# 1. Load audio
signal, sr = load_audio("cough.wav", sr=16000)

# 2. Extract frame-level features
extractor = FeatureExtractor(sr=sr)
frame_features = extractor.extract_features(signal, ["mfcc", "spectral_centroid", "zcr"])

# 3. Aggregate frame-level features into summary statistics
aggregator = FeatureAggregator()
summary = aggregator.aggregate_features(frame_features, ["mean", "std"])

for name, value in summary.items():
    print(f"{name}: {np.asarray(value).shape}")
