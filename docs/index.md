# AudioFeatures 文档

AudioFeatures 是一个轻量级的 Python 音频处理工具包，覆盖音频加载、预处理、特征提取、
增强与可视化。它基于 NumPy、SciPy 与 librosa，提供了易用的 API，便于在研究与工程中快速
搭建音频特征管线。

## 文档导航

- [安装与环境](installation.md)
- [发布指南](publishing.md)
- [快速上手](quickstart.md)
- [使用指南](guide.md)
- [API 概览](api.md)
- [API 参考（自动）](reference.md)
- [常见问题](faq.md)

## 命名说明

- 发行名: `audio_features`（pip 中 `audio_features` 与 `audio-features` 均可）
- 导入名: `audiofeatures`

## 模块概览

- `audiofeatures.core`: 音频加载、分帧与窗函数
- `audiofeatures.preprocessing`: 滤波、归一化与分段
- `audiofeatures.features`: 时域、频域、谱特征与统计特征
- `audiofeatures.augmentation`: 时域与频域增强
- `audiofeatures.pipeline`: 特征提取与聚合流程
- `audiofeatures.utils`: 频率/音符转换与 I/O
- `audiofeatures.visualization`: 常用图表绘制
- `audiofeatures.utils.contract`: float32 与特征矩阵约定

## 快速示例

```python
from audiofeatures.utils import load_audio
from audiofeatures.preprocessing import normalize_amplitude
from audiofeatures.features import mfcc, spectral_centroid

signal, sr = load_audio("example.wav", sr=16000)
signal = normalize_amplitude(signal, target_dBFS=-20.0)

mfccs = mfcc(signal, sr=sr, n_mfcc=13)
centroid = spectral_centroid(signal, sr=sr)
print(mfccs.shape, centroid.shape)
```

## 契约说明

- 音频输入统一 `float32`（推荐范围 `[-1, 1]`）
- 帧级特征统一输出 `(n_frames, n_features)`

## 文档构建

- 安装依赖: `python -m pip install -e ".[docs]"`
- 本地预览: `mkdocs serve`
- 构建静态站点: `mkdocs build`
- 发布到 GitHub Pages: `mkdocs gh-deploy`
