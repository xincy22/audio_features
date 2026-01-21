# API 参考

以下 API 以模块分类，描述主要函数/类、参数含义与返回形状。
除非特别说明，帧级特征统一输出 ``(n_frames, n_features)``，dtype 为 ``float32``。

## audiofeatures.core

### load_audio(file_path, sr=None, mono=True, offset=0.0, duration=None)

- 读取音频文件，返回 `(signal, sr)`
- `sr=None` 表示保持原采样率
- `mono=False` 时返回形状 `(channels, samples)`
- 输出 dtype 为 `float32`
- `offset` 与 `duration` 以秒为单位

### get_audio_info(file_path)

- 返回字典：`sr`, `channels`, `duration`, `samples`, `bit_depth`, `format`
- MP3 没有位深度信息

### frame_signal(signal, frame_length, hop_length, center=True)

- 输入必须是一维数组
- 返回形状 `(n_frames, frame_length)`
- `center=True` 会在两端补零并通常至少返回一帧；`center=False` 且长度不足一帧时返回空数组

### apply_window(frames, window_type="hann")

- 输入必须是二维数组
- 支持 `hann`, `hamming`, `blackman`, `bartlett`, `kaiser`, `rectangular`

## audiofeatures.preprocessing

### filtering

- `low_pass_filter(signal, sr, cutoff_freq, order=4)`
- `high_pass_filter(signal, sr, cutoff_freq, order=4)`
- `band_pass_filter(signal, sr, low_cutoff, high_cutoff, order=4)`
- `median_filter(signal, kernel_size=3)`

以上滤波器均为巴特沃斯实现，截止频率必须小于奈奎斯特频率。

### normalization

- `normalize_amplitude(signal, target_dBFS=-20.0)`：目标 dBFS 归一化
- `peak_normalize(signal, target_peak=0.95)`：峰值归一化
- `z_normalize(signal)`：Z-score 标准化
- `min_max_normalize(signal, min_val=0.0, max_val=1.0)`：最小-最大归一化

输入需为一维数组。RMS/峰值/标准差接近 0 时会返回原始信号并发出警告。

`min_max_normalize` 在近似常数信号上返回常数数组并发出警告。

### segmentation

- `segment_by_energy(signal, sr, threshold=0.05, min_length=0.1)`
- `segment_by_zcr(signal, sr, threshold=0.2, min_length=0.1, frame_length=0.025, hop_length=0.010)`

`segment_by_zcr` 的 `frame_length` 与 `hop_length` 以秒为单位。
返回 `[(start_idx, end_idx), ...]`，单位为采样点索引。

## audiofeatures.features

### time_domain

- `zero_crossing_rate(signal, frame_length=2048, hop_length=512)` -> `(n_frames, 1)`
- `energy(signal, frame_length=2048, hop_length=512)` -> `(n_frames, 1)`
- `log_energy(signal, frame_length=2048, hop_length=512, eps=1e-10)` -> `(n_frames, 1)`
- `pitch(signal, sr, frame_length=2048, hop_length=512, method="autocorr")` -> `(n_frames, 1)`

### frequency_domain

- `magnitude_spectrum(signal, n_fft=2048, hop_length=512, win_length=None, window="hann", center=True, pad_mode="constant")` -> `(n_frames, 1+n_fft//2)`
- `power_spectrum(...)` -> `(n_frames, 1+n_fft//2)`
- `spectral_centroid(signal, sr, ...)` -> `(n_frames, 1)`
- `spectral_bandwidth(signal, sr, ...)` -> `(n_frames, 1)`
- `spectral_rolloff(signal, sr, ..., roll_percent=0.85)` -> `(n_frames, 1)`

### spectral

- `mfcc(signal, sr, n_mfcc=13, ...)` -> `(n_frames, n_mfcc)`
- `delta_mfcc(mfcc_features, order=1, width=9)` -> `(n_frames, n_mfcc)`
- `mel_spectrogram(signal, sr, n_mels=128, ...)` -> `(n_frames, n_mels)`
- `formant_frequencies(signal, sr, order=12, n_formants=4)` -> `(n_frames, n_formants)`

### statistical

- `signal_statistics(signal, frame_length=2048, hop_length=512)` -> dict of `(n_frames, 1)`
- `spectral_statistics(spectrogram, sr, n_fft=2048)` -> dict of `(n_frames, 1)`
- `harmonic_percussive_ratio(signal, sr, margin=3.0, kernel_size=31)` -> float

## audiofeatures.augmentation

### time_domain

- `time_stretch(signal, sr, rate=1.2)`：时间拉伸
- `pitch_shift(signal, sr, n_steps=4)`：变调
- `add_noise(signal, noise_level=0.005, rng=None, seed=None)`：加噪
- `time_mask(signal, mask_fraction=0.1, rng=None, seed=None)`：时间掩码

### frequency_domain

- `spectral_contrast(signal, sr, enhancement_factor=5.0, n_fft=2048, hop_length=512)`
- `harmonic_enhancement(signal, sr, enhancement_factor=2.0)`
- `spectral_inversion(signal)`：反相
- `frequency_mask(signal, sr, mask_start, mask_width, n_fft=2048, hop_length=512)`

## audiofeatures.pipeline

### FeatureExtractor

- `extract_features(signal, feature_types)`：输出 ``(n_frames, n_features)``
  - 支持 `mfcc`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`,
    `zcr`, `rms`, `chroma`, `tonnetz`, `tempogram`
- `extract_from_file(file_path, feature_types)`
- `extract_all_features(signal)`

### FeatureAggregator

- `aggregate_features(features, aggregation_methods)`
  - 支持 `mean`, `std`, `min`, `max`, `median`, `skewness`, `kurtosis`,
    `range`, `quantile_25`, `quantile_75`
- `aggregate_statistics(features)`：返回分组后的统计字典

## audiofeatures.utils

### conversion

- `hz_to_mel(frequencies)` / `mel_to_hz(mels)`
- `hz_to_note(frequency)` / `note_to_hz(note)`
- `seconds_to_samples(seconds, sr)` / `samples_to_seconds(samples, sr)`

### io

- `load_audio(file_path, sr=None, mono=True)`：包装 core.load_audio
- `save_audio(signal, sr, file_path)`：保存为音频文件
- `save_features(features, file_path)` / `load_features(file_path)`：保存/读取 npz

### contract

- `ensure_float32(signal, clip=False)`：转换为 float32，可选裁剪到 [-1, 1]
- `to_feature_matrix(values, frame_axis=0)`：统一为 ``(n_frames, n_features)``

## audiofeatures.visualization

需要安装 `matplotlib`。

- `plot_waveform(signal, sr, ...)`
- `plot_energy(signal, sr, ...)`
- `plot_zero_crossing_rate(signal, sr, ...)`
- `plot_spectrogram(signal, sr, ...)`
- `plot_mel_spectrogram(signal, sr, ...)`
- `plot_mfcc(signal, sr, ...)`
- `plot_chromagram(signal, sr, ...)`
