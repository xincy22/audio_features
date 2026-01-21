# 使用指南

本指南以完整流程为主线：加载 -> 预处理 -> 特征 -> 聚合 -> 保存 -> 可视化/增强。

## 1. 加载音频

```python
from audiofeatures.utils import load_audio

signal, sr = load_audio("example.wav", sr=16000, mono=True)
```

`load_audio` 默认返回一维信号。如果你想做更细的音频信息读取，使用：

```python
from audiofeatures.core import get_audio_info

info = get_audio_info("example.wav")
print(info["sr"], info["duration"])
```

## 2. 预处理

常见流程：滤波 + 归一化 + 分段。

```python
from audiofeatures.preprocessing import (
    low_pass_filter,
    normalize_amplitude,
    segment_by_energy,
)

filtered = low_pass_filter(signal, sr, cutoff_freq=4000)
normalized = normalize_amplitude(filtered, target_dBFS=-20.0)
segments = segment_by_energy(normalized, sr, threshold=0.1, min_length=0.2)
```

`segment_by_energy` 与 `segment_by_zcr` 返回采样点索引区间，可用于裁剪：

```python
for start, end in segments:
    clip = normalized[start:end]
```

## 3. 手动提取特征

### 时域特征

```python
from audiofeatures.features import zero_crossing_rate, energy, pitch

zcr = zero_crossing_rate(signal, frame_length=2048, hop_length=512)
eng = energy(signal, frame_length=2048, hop_length=512)
pitches = pitch(signal, sr=sr, frame_length=1024, hop_length=512)
```

### 频域与谱特征

```python
from audiofeatures.features import (
    magnitude_spectrum,
    spectral_centroid,
    mfcc,
    mel_spectrogram,
)

mag = magnitude_spectrum(signal, n_fft=2048, hop_length=512)
centroid = spectral_centroid(signal, sr=sr)
mfccs = mfcc(signal, sr=sr, n_mfcc=13)
mel = mel_spectrogram(signal, sr=sr)
```

## 4. 使用 Pipeline

`FeatureExtractor` 可以一次提取多种特征，`FeatureAggregator` 负责聚合为固定长度向量。

```python
from audiofeatures.pipeline import FeatureExtractor, FeatureAggregator

extractor = FeatureExtractor(sr=16000)
frame_features = extractor.extract_features(
    signal,
    ["mfcc", "spectral_centroid", "zcr"]
)

aggregator = FeatureAggregator()
summary = aggregator.aggregate_features(frame_features, ["mean", "std"])
```

聚合后的 `summary` 是字典，键会带上统计方式后缀，例如 `mfcc_mean`。

## 5. 保存与加载特征

```python
from audiofeatures.utils import save_features, load_features

save_features(summary, "features.npz")
features = load_features("features.npz")
```

## 6. 数据增强

```python
from audiofeatures.augmentation import time_stretch, pitch_shift, add_noise

stretched = time_stretch(signal, sr=sr, rate=1.2)
shifted = pitch_shift(signal, sr=sr, n_steps=3)
noisy = add_noise(signal, noise_level=0.005)
```

## 7. 可视化

```python
from audiofeatures.visualization import plot_waveform, plot_spectrogram

fig = plot_waveform(signal, sr=sr, title="Waveform")
fig.savefig("waveform.png")

fig = plot_spectrogram(signal, sr=sr, title="Spectrogram")
fig.savefig("spectrogram.png")
```

## 8. 参数选择建议

- `frame_length` 与 `hop_length` 通常使用 2048/512 或 1024/256 组合
- `sr` 建议与模型或任务需求一致
- 归一化在特征提取前完成更稳妥
