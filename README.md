# AudioFeatures

[English](README.md) | [中文](README.zh-CN.md)

AudioFeatures is a compact Python toolkit for audio preprocessing, feature extraction, augmentation, and visualization. It exposes a clean API built on top of NumPy, SciPy, and librosa so you can go from waveform to features quickly.

## Highlights

- Audio loading and metadata utilities
- Preprocessing: filtering, normalization, segmentation
- Time- and frequency-domain features
- Spectral features (MFCC, mel spectrogram, formants)
- Data augmentation helpers
- Feature aggregation for modeling
- Visualization helpers for common plots

## Installation

Supported Python: 3.9 - 3.13.

```bash
pip install audio_features
```

Optional visualization dependencies:

```bash
pip install "audio_features[viz]"
```

Import name remains `audiofeatures`.

## Quickstart

```python
from audiofeatures.utils import load_audio
from audiofeatures.features import mfcc, spectral_centroid
from audiofeatures.preprocessing import normalize_amplitude

signal, sr = load_audio("example.wav", sr=16000)
signal = normalize_amplitude(signal, target_dBFS=-20.0)

mfccs = mfcc(signal, sr=sr, n_mfcc=13)
centroid = spectral_centroid(signal, sr=sr)

print(mfccs.shape, centroid.shape)
```

## Pipeline Example

```python
from audiofeatures.utils import load_audio
from audiofeatures.pipeline import FeatureExtractor, FeatureAggregator

signal, sr = load_audio("example.wav", sr=16000)
extractor = FeatureExtractor(sr=sr)
frame_features = extractor.extract_features(signal, ["mfcc", "spectral_centroid", "zcr"])

aggregator = FeatureAggregator()
summary = aggregator.aggregate_features(frame_features, ["mean", "std"])
print(summary.keys())
```

## Notes

- MP3 decoding depends on system backends (e.g., ffmpeg). If MP3 loading fails, install ffmpeg or use WAV/FLAC inputs.
- Visualization helpers require `matplotlib`.

## Documentation

See `docs/index.md` for the full guide and API reference.

## Tests

```bash
python -m unittest discover -s tests
```

## License

MIT License. See LICENSE.
