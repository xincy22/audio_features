"""时域数据增强函数。"""

import numpy as np
import librosa


def time_stretch(signal, sr, rate=1.2):
    """对信号进行时间拉伸（不改变音高）。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    rate : float, optional
        拉伸倍率，>1 加速，<1 放慢。

    Returns
    -------
    ndarray
        拉伸后的信号。

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
    if rate <= 0:
        raise ValueError("rate must be > 0")

    return librosa.effects.time_stretch(signal, rate=rate)


def pitch_shift(signal, sr, n_steps=4):
    """对信号进行变调（不改变时长）。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    n_steps : float, optional
        变化的半音数，正值升高、负值降低。

    Returns
    -------
    ndarray
        变调后的信号。

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

    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)


def add_noise(signal, noise_level=0.005):
    """向信号添加高斯白噪声。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    noise_level : float, optional
        噪声标准差。

    Returns
    -------
    ndarray
        加噪后的信号。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if noise_level < 0:
        raise ValueError("noise_level must be >= 0")

    noise = np.random.randn(signal.size) * noise_level
    return signal + noise


def time_mask(signal, mask_fraction=0.1):
    """对信号随机时间片段进行掩码。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    mask_fraction : float, optional
        掩码比例，范围 [0, 1]。

    Returns
    -------
    ndarray
        掩码后的信号。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if not 0 <= mask_fraction <= 1:
        raise ValueError("mask_fraction must be in [0, 1]")

    masked = signal.copy()
    mask_len = int(round(masked.size * mask_fraction))
    if mask_len == 0:
        return masked
    start = np.random.randint(0, max(1, masked.size - mask_len + 1))
    masked[start:start + mask_len] = 0.0
    return masked
