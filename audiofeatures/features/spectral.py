"""谱特征提取函数。"""

import numpy as np
import librosa

from audiofeatures.core.signal_processing import frame_signal


def mfcc(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128, fmin=0.0, fmax=None):
    """计算 MFCC（梅尔频率倒谱系数）。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    n_mfcc : int, optional
        MFCC 系数数量。
    n_fft : int, optional
        FFT 点数。
    hop_length : int, optional
        帧移（样本数）。
    n_mels : int, optional
        Mel 滤波器组数量。
    fmin : float, optional
        最低频率（Hz）。
    fmax : float or None, optional
        最高频率（Hz），默认 ``sr / 2``。

    Returns
    -------
    ndarray
        MFCC 特征，形状为 ``(n_mfcc, n_frames)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。

    Notes
    -----
    MFCC 常用于语音识别、说话人识别等任务。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    return librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )


def delta_mfcc(mfcc_features, order=1, width=9):
    """计算 MFCC 的差分特征。

    Parameters
    ----------
    mfcc_features : ndarray
        MFCC 特征，形状为 ``(n_mfcc, n_frames)``。
    order : int, optional
        差分阶数：1 为一阶，2 为二阶。
    width : int, optional
        差分窗口宽度（奇数）。

    Returns
    -------
    ndarray
        差分后的 MFCC，形状与输入一致。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。
    """
    mfcc_features = np.asarray(mfcc_features)
    if mfcc_features.ndim != 2:
        raise ValueError("mfcc_features must be a 2D array")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if width < 3 or width % 2 == 0:
        raise ValueError("width must be an odd integer >= 3")

    return librosa.feature.delta(mfcc_features, order=order, width=width)


def mel_spectrogram(signal, sr, n_fft=2048, hop_length=512, n_mels=128, fmin=0.0, fmax=None):
    """计算 Mel 频谱。

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
    n_mels : int, optional
        Mel 滤波器组数量。
    fmin : float, optional
        最低频率（Hz）。
    fmax : float or None, optional
        最高频率（Hz），默认 ``sr / 2``。

    Returns
    -------
    ndarray
        Mel 频谱，形状为 ``(n_mels, n_frames)``。

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

    return librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )


def formant_frequencies(signal, sr, order=12, n_formants=4):
    """估计共振峰频率（LPC）。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    order : int, optional
        LPC 阶数。
    n_formants : int, optional
        估计的共振峰数量。

    Returns
    -------
    ndarray
        共振峰频率数组，形状为 ``(n_frames, n_formants)``。

    Raises
    ------
    ValueError
        输入维度或参数非法时抛出。

    Notes
    -----
    LPC 共振峰估计对噪声敏感，结果仅供近似分析。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if order <= 0:
        raise ValueError("order must be > 0")
    if n_formants <= 0:
        raise ValueError("n_formants must be > 0")

    frame_length = max(3, int(0.03 * sr))
    hop_length = max(1, int(0.01 * sr))
    frames = frame_signal(signal, frame_length=frame_length, hop_length=hop_length, center=True)
    if frames.size == 0:
        return np.empty((0, n_formants))

    formants = np.zeros((frames.shape[0], n_formants), dtype=float)
    for i, frame in enumerate(frames):
        frame = frame - np.mean(frame)
        if np.allclose(frame, 0.0):
            continue
        try:
            lpc_coeffs = librosa.lpc(frame, order=order)
        except Exception:
            continue
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2.0 * np.pi))
        freqs = np.sort(freqs[(freqs > 0) & (freqs < sr / 2.0)])
        if freqs.size:
            formants[i, : min(n_formants, freqs.size)] = freqs[:n_formants]
    return formants
