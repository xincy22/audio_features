"""统计特征提取函数。"""

import numpy as np
from scipy import stats
import librosa

from audiofeatures.core.signal_processing import frame_signal
from audiofeatures.utils.contract import ensure_float32, to_feature_matrix


def signal_statistics(signal, frame_length=2048, hop_length=512):
    """计算信号的帧级统计量。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    frame_length : int, optional
        帧长度（样本数）。
    hop_length : int, optional
        帧移（样本数）。

    Returns
    -------
    dict
        统计量字典，包含 ``mean``、``std``、``skewness``、``kurtosis``、
        ``median``、``min``、``max``、``range``、``rms``，每项形状为
        ``(n_frames, 1)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if frame_length <= 0:
        raise ValueError("frame_length must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    frames = frame_signal(signal, frame_length=frame_length, hop_length=hop_length, center=True)
    if frames.size == 0:
        empty = np.zeros((0, 1), dtype=np.float32)
        return {
            "mean": empty,
            "std": empty,
            "skewness": empty,
            "kurtosis": empty,
            "median": empty,
            "min": empty,
            "max": empty,
            "range": empty,
            "rms": empty
        }

    mean = np.mean(frames, axis=1)
    std = np.std(frames, axis=1)
    frames_stats = frames.astype(np.float64, copy=False)
    skewness = stats.skew(frames_stats, axis=1, bias=False)
    kurtosis = stats.kurtosis(frames_stats, axis=1, bias=False)
    median = np.median(frames, axis=1)
    min_val = np.min(frames, axis=1)
    max_val = np.max(frames, axis=1)
    range_val = max_val - min_val
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    return {
        "mean": to_feature_matrix(mean),
        "std": to_feature_matrix(std),
        "skewness": to_feature_matrix(skewness),
        "kurtosis": to_feature_matrix(kurtosis),
        "median": to_feature_matrix(median),
        "min": to_feature_matrix(min_val),
        "max": to_feature_matrix(max_val),
        "range": to_feature_matrix(range_val),
        "rms": to_feature_matrix(rms)
    }


def spectral_statistics(spectrogram, sr, n_fft=2048):
    """计算频谱的统计特征。

    Parameters
    ----------
    spectrogram : ndarray
        频谱矩阵，形状为 ``(n_frames, 1 + n_fft // 2)``。
    sr : int
        采样率（Hz）。
    n_fft : int, optional
        计算频谱时使用的 FFT 点数。

    Returns
    -------
    dict
        频谱统计量字典，包含 ``centroid``、``bandwidth``、``flatness``、
        ``rolloff``、``flux``、``contrast``，每项形状为 ``(n_frames, 1)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    spectrogram = to_feature_matrix(spectrogram, frame_axis=0)
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")

    spec = spectrogram.T
    centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=sr, n_fft=n_fft)
    flatness = librosa.feature.spectral_flatness(S=spec)
    rolloff = librosa.feature.spectral_rolloff(S=spec, sr=sr)

    flux = np.sqrt(np.sum(np.diff(spectrogram, axis=0) ** 2, axis=1))
    flux = np.concatenate(([0.0], flux))

    contrast = librosa.feature.spectral_contrast(S=spec, sr=sr)
    contrast_mean = np.mean(contrast, axis=0)

    return {
        "centroid": to_feature_matrix(centroid, frame_axis=1),
        "bandwidth": to_feature_matrix(bandwidth, frame_axis=1),
        "flatness": to_feature_matrix(flatness, frame_axis=1),
        "rolloff": to_feature_matrix(rolloff, frame_axis=1),
        "flux": to_feature_matrix(flux),
        "contrast": to_feature_matrix(contrast_mean)
    }


def harmonic_percussive_ratio(signal, sr, margin=3.0, kernel_size=31):
    """估计谐波-打击乐能量比。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    margin : float, optional
        HPSS 分离的裕度参数。
    kernel_size : int, optional
        中值滤波器核大小。

    Returns
    -------
    float
        谐波能量占比，范围 ``[0, 1]``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    stft = librosa.stft(signal)
    harmonic, percussive = librosa.decompose.hpss(
        stft,
        margin=margin,
        kernel_size=kernel_size
    )
    harmonic_energy = np.sum(np.abs(harmonic) ** 2)
    percussive_energy = np.sum(np.abs(percussive) ** 2)
    total = harmonic_energy + percussive_energy
    if total == 0:
        return 0.0
    return float(harmonic_energy / total)
