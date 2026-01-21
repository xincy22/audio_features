"""
基本使用示例
============

这个示例展示了AudioFeatures库的基本用法。
"""

from audiofeatures.utils import load_audio
from audiofeatures.features import mfcc, spectral_centroid, zero_crossing_rate
from audiofeatures.preprocessing import normalize_amplitude
from audiofeatures.visualization import plot_waveform, plot_spectrogram, plot_mfcc

# 加载音频文件
signal, sr = load_audio("example.wav", sr=22050)
print(f"音频长度: {len(signal)/sr:.2f}秒, 采样率: {sr}Hz")

# 预处理: 归一化振幅
normalized_signal = normalize_amplitude(signal, target_dBFS=-20.0)

# 提取特征
mfccs = mfcc(normalized_signal, sr=sr, n_mfcc=13)
centroid = spectral_centroid(normalized_signal, sr=sr)
zcr = zero_crossing_rate(normalized_signal)

print(f"MFCC形状: {mfccs.shape}")
print(f"谱质心形状: {centroid.shape}")
print(f"过零率形状: {zcr.shape}")

# 可视化
# 波形图
fig_wave = plot_waveform(normalized_signal, sr=sr, title="Waveform")
fig_wave.savefig("waveform.png")

# 频谱图
fig_spec = plot_spectrogram(normalized_signal, sr=sr, title="Spectrogram")
fig_spec.savefig("spectrogram.png")

# MFCC
fig_mfcc = plot_mfcc(normalized_signal, sr=sr, title="MFCC")
fig_mfcc.savefig("mfcc.png")

print("示例完成，图像已保存。")
