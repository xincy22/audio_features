"""时域特征提取函数。"""

import numpy as np

from audiofeatures.core.signal_processing import frame_signal


def zero_crossing_rate(signal, frame_length=2048, hop_length=512):
    """计算过零率（ZCR）。

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
    ndarray
        每帧过零率，形状为 ``(n_frames,)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。

    Notes
    -----
    过零率对噪声较敏感，建议配合降噪或预处理。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if frame_length < 2:
        raise ValueError("frame_length must be >= 2")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    frames = frame_signal(signal, frame_length=frame_length, hop_length=hop_length, center=False)
    signs = np.signbit(frames)
    crossings = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    return crossings / (frame_length - 1)


def energy(signal, frame_length=2048, hop_length=512):
    """计算短时能量。

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
    ndarray
        每帧能量，形状为 ``(n_frames,)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if frame_length <= 0:
        raise ValueError("frame_length must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    frames = frame_signal(signal, frame_length=frame_length, hop_length=hop_length, center=False)
    return np.sum(frames ** 2, axis=1)


def log_energy(signal, frame_length=2048, hop_length=512, eps=1e-10):
    """计算对数短时能量。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    frame_length : int, optional
        帧长度（样本数）。
    hop_length : int, optional
        帧移（样本数）。
    eps : float, optional
        防止 ``log(0)`` 的小常数。

    Returns
    -------
    ndarray
        每帧对数能量，形状为 ``(n_frames,)``。
    """
    energy_values = energy(signal, frame_length=frame_length, hop_length=hop_length)
    return np.log(energy_values + eps)


def pitch(signal, sr, frame_length=2048, hop_length=512, method="autocorr"):
    """估计基频（音高）。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    frame_length : int, optional
        帧长度（样本数）。
    hop_length : int, optional
        帧移（样本数）。
    method : {'autocorr', 'improved_autocorr'}, optional
        基于自相关的两种估计方式。

    Returns
    -------
    ndarray
        每帧基频（Hz），形状为 ``(n_frames,)``。

    Raises
    ------
    ValueError
        输入非法或方法不支持时抛出。

    Notes
    -----
    自相关法在噪声或多音情况下可能不稳定。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if frame_length <= 0:
        raise ValueError("frame_length must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    if method not in {"autocorr", "improved_autocorr"}:
        raise ValueError("method must be 'autocorr' or 'improved_autocorr'")

    frames = frame_signal(signal, frame_length=frame_length, hop_length=hop_length, center=False)
    if frames.size == 0:
        return np.array([])

    min_freq = 50.0
    max_freq = min(2000.0, sr / 2.0)
    min_lag = max(1, int(sr / max_freq))
    max_lag = min(frames.shape[1] - 1, int(sr / min_freq))
    window = np.hanning(frame_length)

    pitches = np.zeros(frames.shape[0], dtype=float)
    for i, frame in enumerate(frames):
        frame = frame - np.mean(frame)
        if method == "improved_autocorr":
            frame = frame * window
        if np.allclose(frame, 0.0):
            pitches[i] = 0.0
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_length - 1:]
        if max_lag <= min_lag:
            pitches[i] = 0.0
            continue
        corr[:min_lag] = 0.0
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        pitches[i] = sr / lag if lag > 0 else 0.0
    return pitches
