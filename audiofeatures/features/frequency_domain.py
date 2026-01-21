"""频域特征提取函数。"""

import numpy as np
import librosa

from audiofeatures.utils.contract import ensure_float32, to_feature_matrix


def magnitude_spectrum(
    signal,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant"
):
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
    center : bool, optional
        是否在帧中心对齐。
    pad_mode : str, optional
        边界填充模式，透传给 ``librosa.stft``。

    Returns
    -------
    ndarray
        幅度谱，形状为 ``(n_frames, 1 + n_fft // 2)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = ensure_float32(signal)
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
        center=center,
        pad_mode=pad_mode
    )
    magnitude = np.abs(stft)
    return to_feature_matrix(magnitude, frame_axis=1)


def power_spectrum(
    signal,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant"
):
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
    center : bool, optional
        是否在帧中心对齐。
    pad_mode : str, optional
        边界填充模式。

    Returns
    -------
    ndarray
        功率谱，形状为 ``(n_frames, 1 + n_fft // 2)``。
    """
    magnitude = magnitude_spectrum(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode
    )
    power = magnitude ** 2
    return power.astype(np.float32, copy=False)


def spectral_centroid(
    signal,
    sr,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant"
):
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
    center : bool, optional
        是否在帧中心对齐。
    pad_mode : str, optional
        边界填充模式。

    Returns
    -------
    ndarray
        谱质心序列（Hz），形状为 ``(n_frames, 1)``。

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
        window=window,
        center=center,
        pad_mode=pad_mode
    )
    return to_feature_matrix(centroid, frame_axis=1)


def spectral_bandwidth(
    signal,
    sr,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    p=2,
    center=True,
    pad_mode="constant"
):
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
    center : bool, optional
        是否在帧中心对齐。
    pad_mode : str, optional
        边界填充模式。

    Returns
    -------
    ndarray
        谱带宽序列（Hz），形状为 ``(n_frames, 1)``。

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
        p=p,
        center=center,
        pad_mode=pad_mode
    )
    return to_feature_matrix(bandwidth, frame_axis=1)


def spectral_rolloff(
    signal,
    sr,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    roll_percent=0.85,
    center=True,
    pad_mode="constant"
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
    center : bool, optional
        是否在帧中心对齐。
    pad_mode : str, optional
        边界填充模式。

    Returns
    -------
    ndarray
        谱滚降点序列（Hz），形状为 ``(n_frames, 1)``。

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
        roll_percent=roll_percent,
        center=center,
        pad_mode=pad_mode
    )
    return to_feature_matrix(rolloff, frame_axis=1)
