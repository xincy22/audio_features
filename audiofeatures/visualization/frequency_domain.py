"""频域可视化工具。"""

import numpy as np
import librosa
import librosa.display


def plot_spectrogram(signal, sr, n_fft=2048, hop_length=512, title="Spectrogram", figsize=(10, 6), ax=None):
    """绘制线性频率谱图。

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
    title : str, optional
        图表标题。
    figsize : tuple, optional
        图表大小（英寸）。
    ax : matplotlib.axes.Axes or None, optional
        可选的坐标轴对象。

    Returns
    -------
    matplotlib.figure.Figure
        绘制后的图对象。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    import matplotlib.pyplot as plt

    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    db = librosa.amplitude_to_db(magnitude, ref=np.max)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig


def plot_mel_spectrogram(
    signal,
    sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    title="Mel Spectrogram",
    figsize=(10, 6),
    ax=None
):
    """绘制 Mel 频谱图。

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
    title : str, optional
        图表标题。
    figsize : tuple, optional
        图表大小（英寸）。
    ax : matplotlib.axes.Axes or None, optional
        可选的坐标轴对象。

    Returns
    -------
    matplotlib.figure.Figure
        绘制后的图对象。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    import matplotlib.pyplot as plt

    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig


def plot_mfcc(signal, sr, n_mfcc=13, title="MFCC", figsize=(10, 6), ax=None):
    """绘制 MFCC 特征图。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    n_mfcc : int, optional
        MFCC 系数数量。
    title : str, optional
        图表标题。
    figsize : tuple, optional
        图表大小（英寸）。
    ax : matplotlib.axes.Axes or None, optional
        可选的坐标轴对象。

    Returns
    -------
    matplotlib.figure.Figure
        绘制后的图对象。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    import matplotlib.pyplot as plt

    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    img = librosa.display.specshow(mfccs, x_axis="time", ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax)
    return fig


def plot_chromagram(signal, sr, title="Chromagram", figsize=(10, 6), ax=None):
    """绘制色度图。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    title : str, optional
        图表标题。
    figsize : tuple, optional
        图表大小（英寸）。
    ax : matplotlib.axes.Axes or None, optional
        可选的坐标轴对象。

    Returns
    -------
    matplotlib.figure.Figure
        绘制后的图对象。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    import matplotlib.pyplot as plt

    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    img = librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax)
    return fig
