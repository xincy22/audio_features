# AudioFeatures

[English](README.md) | [中文](README.zh-CN.md)

AudioFeatures 是一个轻量级的 Python 音频工具包，覆盖音频预处理、特征提取、增强与可视化。基于 NumPy、SciPy 与 librosa，提供简洁易用的 API，方便你快速从波形获取特征。

## 亮点

- 音频加载与元数据工具
- 预处理：滤波、归一化、分段
- 时域与频域特征
- 频谱特征（MFCC、Mel 频谱、共振峰）
- 数据增强工具
- 面向建模的特征聚合
- 常见图形的可视化辅助

## 安装

支持 Python: 3.9 - 3.13。

```bash
pip install audio-features
# 或
pip install audio_features
```

可选的可视化依赖：

```bash
pip install "audio_features[viz]"
```

导入包名仍为 `audiofeatures`。

## 快速上手

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

帧级特征统一输出 `(n_frames, n_features)`，dtype 为 `float32`。

## 流水线示例

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

## 说明

- MP3 解码依赖系统后端（例如 ffmpeg）。若加载 MP3 失败，请安装 ffmpeg 或改用 WAV/FLAC。
- 可视化功能需要 `matplotlib`。

## 文档

完整使用与 API 参考见 `docs/index.md`。

## 测试

```bash
python -m pytest
```

## 许可证

MIT License。详见 LICENSE。
