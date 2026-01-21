"""时域可视化工具。"""

import numpy as np

from audiofeatures.features.time_domain import energy as energy_feature
from audiofeatures.features.time_domain import zero_crossing_rate as zcr_feature


def plot_waveform(signal, sr, title="Waveform", figsize=(10, 4), ax=None):
    """绘制波形图。

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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    time = np.arange(signal.size) / float(sr)
    ax.plot(time, signal, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig


def plot_energy(signal, sr, frame_length=2048, hop_length=512, title="Signal Energy", figsize=(10, 4), ax=None):
    """绘制短时能量曲线。

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

    energy_values = energy_feature(signal, frame_length=frame_length, hop_length=hop_length)
    times = np.arange(energy_values.size) * (hop_length / float(sr))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(times, energy_values, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    return fig


def plot_zero_crossing_rate(
    signal,
    sr,
    frame_length=2048,
    hop_length=512,
    title="Zero Crossing Rate",
    figsize=(10, 4),
    ax=None
):
    """绘制过零率曲线。

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

    zcr_values = zcr_feature(signal, frame_length=frame_length, hop_length=hop_length)
    times = np.arange(zcr_values.size) * (hop_length / float(sr))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(times, zcr_values, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ZCR")
    return fig
