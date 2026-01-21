"""
音频增强示例
==========

这个示例展示了如何使用AudioFeatures的增强功能来生成更多的训练数据。
"""

import os

from audiofeatures.utils import load_audio, save_audio
from audiofeatures.augmentation import (
    time_stretch,
    pitch_shift,
    add_noise,
    time_mask,
    frequency_mask
)

# 设置参数
input_file = "example.wav"
output_dir = "augmented/"
sr = 22050

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载音频
signal, sr = load_audio(input_file, sr=sr)
print(f"原始音频长度: {len(signal)/sr:.2f}秒")

# 应用不同的增强方法
# 1. 时间拉伸
for rate in [0.8, 1.2]:
    stretched = time_stretch(signal, sr=sr, rate=rate)
    output_path = os.path.join(output_dir, f"time_stretch_{rate}.wav")
    save_audio(stretched, sr, output_path)
    print(f"时间拉伸 (rate={rate}): {len(stretched)/sr:.2f}秒")

# 2. 音高变换
for n_steps in [-3, 3]:
    shifted = pitch_shift(signal, sr=sr, n_steps=n_steps)
    output_path = os.path.join(output_dir, f"pitch_shift_{n_steps}.wav")
    save_audio(shifted, sr, output_path)
    print(f"音高变换 (n_steps={n_steps}): {len(shifted)/sr:.2f}秒")

# 3. 添加噪声
for noise_level in [0.005, 0.01]:
    noisy = add_noise(signal, noise_level=noise_level)
    output_path = os.path.join(output_dir, f"noise_{noise_level}.wav")
    save_audio(noisy, sr, output_path)
    print(f"添加噪声 (level={noise_level})")

# 4. 时间掩码
masked = time_mask(signal, mask_fraction=0.1)
output_path = os.path.join(output_dir, "time_mask.wav")
save_audio(masked, sr, output_path)
print("时间掩码应用完成")

# 5. 频率掩码
freq_masked = frequency_mask(signal, sr=sr, mask_start=1000, mask_width=1000)
output_path = os.path.join(output_dir, "freq_mask.wav")
save_audio(freq_masked, sr, output_path)
print("频率掩码应用完成")

print(f"增强音频已保存到 {output_dir}")
