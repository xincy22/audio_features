"""频域特征提取函数。"""

import numpy as np
import librosa


def magnitude_spectrum(signal, n_fft=2048, hop_length=512, win_length=None, window="hann"):
    """计算幅度谱。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。
    win_length : int or None, optional
        窗长度（样本数），默认等于 ``n_fft``。
    window : str, optional
        窗函数类型，传入 ``librosa.stft``。

    Returns
    -------
    ndarray
        幅度谱，形状为 ``(1 + n_fft // 2, n_frames)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    win_length = n_fft if win_length is None else win_length
    stft = librosa.stft(
        y=signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True
    )
    return np.abs(stft)


def power_spectrum(signal, n_fft=2048, hop_length=512, win_length=None, window="hann"):
    """计算功率谱。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。
    win_length : int or None, optional
        窗长度（样本数），默认等于 ``n_fft``。
    window : str, optional
        窗函数类型。

    Returns
    -------
    ndarray
        功率谱，形状为 ``(1 + n_fft // 2, n_frames)``。
    """
    magnitude = magnitude_spectrum(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    return magnitude ** 2


def spectral_centroid(signal, sr, n_fft=2048, hop_length=512, win_length=None, window="hann"):
    """计算谱质心。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。
    win_length : int or None, optional
        窗长度（样本数），默认等于 ``n_fft``。
    window : str, optional
        窗函数类型。

    Returns
    -------
    ndarray
        谱质心序列（Hz）。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    win_length = n_fft if win_length is None else win_length
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    return centroid.flatten()


def spectral_bandwidth(signal, sr, n_fft=2048, hop_length=512, win_length=None, window="hann", p=2):
    """计算谱带宽。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。
    win_length : int or None, optional
        窗长度（样本数），默认等于 ``n_fft``。
    window : str, optional
        窗函数类型。
    p : float, optional
        带宽计算的幂次。

    Returns
    -------
    ndarray
        谱带宽序列（Hz）。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    win_length = n_fft if win_length is None else win_length
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        p=p
    )
    return bandwidth.flatten()


def spectral_rolloff(
    signal,
    sr,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    roll_percent=0.85
):
    """计算谱滚降点。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。
    win_length : int or None, optional
        窗长度（样本数），默认等于 ``n_fft``。
    window : str, optional
        窗函数类型。
    roll_percent : float, optional
        能量累积分位，默认 0.85。

    Returns
    -------
    ndarray
        谱滚降点序列（Hz）。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    win_length = n_fft if win_length is None else win_length
    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        roll_percent=roll_percent
    )
    return rolloff.flatten()
