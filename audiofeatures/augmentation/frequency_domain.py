"""频域数据增强函数。"""

import numpy as np
import librosa


def spectral_contrast(signal, sr, enhancement_factor=5.0, n_fft=2048, hop_length=512):
    """增强谱对比度。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    enhancement_factor : float, optional
        增强系数，数值越大对比越强。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。

    Returns
    -------
    ndarray
        增强后的信号。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if enhancement_factor <= 0:
        raise ValueError("enhancement_factor must be > 0")

    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    log_mag = np.log1p(magnitude)
    mean_log = np.mean(log_mag, axis=0, keepdims=True)
    enhanced_log = mean_log + enhancement_factor * (log_mag - mean_log)
    enhanced_mag = np.maximum(np.expm1(enhanced_log), 0.0)
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    return librosa.istft(enhanced_stft, hop_length=hop_length, length=signal.size)


def harmonic_enhancement(signal, sr, enhancement_factor=2.0):
    """增强谐波成分。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    enhancement_factor : float, optional
        谐波增强倍数。

    Returns
    -------
    ndarray
        增强后的信号。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if enhancement_factor <= 0:
        raise ValueError("enhancement_factor must be > 0")

    harmonic = librosa.effects.harmonic(signal)
    enhanced = signal + (enhancement_factor - 1.0) * harmonic
    max_abs = np.max(np.abs(enhanced))
    if max_abs > 1.0:
        enhanced = enhanced / max_abs
    return enhanced


def spectral_inversion(signal):
    """对信号进行极性反转。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。

    Returns
    -------
    ndarray
        反相后的信号。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    return -signal


def frequency_mask(signal, sr, mask_start, mask_width, n_fft=2048, hop_length=512):
    """对频谱的指定频带进行掩码。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    mask_start : float
        掩码起始频率（Hz）。
    mask_width : float
        掩码频带宽度（Hz）。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。

    Returns
    -------
    ndarray
        频带掩码后的信号。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if mask_start < 0 or mask_width <= 0:
        raise ValueError("mask_start must be >= 0 and mask_width must be > 0")

    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask_end = mask_start + mask_width
    mask = (freqs >= mask_start) & (freqs <= mask_end)
    stft[mask, :] = 0.0
    return librosa.istft(stft, hop_length=hop_length, length=signal.size)
