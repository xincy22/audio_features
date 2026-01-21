# 快速上手

下面示例展示了从加载音频到提取特征的最短路径。

```python
from audiofeatures.utils import load_audio
from audiofeatures.preprocessing import normalize_amplitude
from audiofeatures.features import mfcc, spectral_centroid, zero_crossing_rate

signal, sr = load_audio("example.wav", sr=16000)
signal = normalize_amplitude(signal, target_dBFS=-20.0)

mfccs = mfcc(signal, sr=sr, n_mfcc=13)
centroid = spectral_centroid(signal, sr=sr)
zcr = zero_crossing_rate(signal)

print(mfccs.shape, centroid.shape, zcr.shape)
```

如果你想一次性提取多个特征并聚合为固定长度向量，请参考 [使用指南](guide.md)。
