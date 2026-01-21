"""
Batch feature extraction example.
"""

import numpy as np

from audiofeatures.utils import load_audio
from audiofeatures.pipeline import FeatureExtractor, FeatureAggregator

audio_files = ["cough1.wav", "cough2.wav", "cough3.wav"]
extractor = FeatureExtractor(sr=16000)
aggregator = FeatureAggregator()

rows = []
for path in audio_files:
    signal, _ = load_audio(path, sr=extractor.sr)
    frame_features = extractor.extract_features(signal, ["mfcc", "spectral_centroid", "zcr"])
    summary = aggregator.aggregate_features(frame_features, ["mean", "std"])
    flat = {key: np.asarray(value).flatten() for key, value in summary.items()}
    rows.append(flat)

print(f"Processed {len(rows)} files")
