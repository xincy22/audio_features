# 常见问题

## 1. MP3 无法加载或报错

`librosa` 依赖系统解码后端。请安装 ffmpeg，或将音频转换为 WAV/FLAC。

## 2. 返回空数组或空列表

当信号长度小于 `frame_length` 时，分帧类函数会返回空结果。检查音频长度，
或降低 `frame_length`。

## 3. 为什么所有特征都是二维

为了避免下游处理混乱，所有帧级特征统一输出 `(n_frames, n_features)`，即使只有一个
维度也会返回 `(n_frames, 1)`。这样在拼接或聚合时更稳定。

## 4. 过零率/能量结果和预期差异较大

确保输入为一维数组且已归一化。必要时先滤波降噪，再提取特征。

## 5. 可视化模块找不到 matplotlib

执行：

```bash
pip install "audio_features[viz]"
```

## 6. 发行名与导入名不一致

发行名是 `audio_features`（pip 中也可以写 `audio-features`），
导入名仍然是 `audiofeatures`。
