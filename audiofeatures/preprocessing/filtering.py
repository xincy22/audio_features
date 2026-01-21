"""音频信号滤波工具。"""

import numpy as np
from scipy import signal as sci_signal


def low_pass_filter(signal, sr, cutoff_freq, order=4):
    """应用巴特沃斯低通滤波。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    cutoff_freq : float
        截止频率（Hz）。
    order : int, optional
        滤波器阶数。

    Returns
    -------
    ndarray
        低通滤波后的信号。

    Raises
    ------
    ValueError
        参数非法或超过奈奎斯特频率时抛出。
    """
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if cutoff_freq <= 0:
        raise ValueError("cutoff_freq must be > 0")
    if order <= 0:
        raise ValueError("order must be > 0")
    nyquist = 0.5 * sr
    if cutoff_freq >= nyquist:
        raise ValueError("cutoff_freq must be less than Nyquist frequency")
    normal_cutoff = cutoff_freq / nyquist
    b, a = sci_signal.butter(order, normal_cutoff, btype="low", analog=False)
    return sci_signal.filtfilt(b, a, signal)


def high_pass_filter(signal, sr, cutoff_freq, order=4):
    """应用巴特沃斯高通滤波。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    cutoff_freq : float
        截止频率（Hz）。
    order : int, optional
        滤波器阶数。

    Returns
    -------
    ndarray
        高通滤波后的信号。

    Raises
    ------
    ValueError
        参数非法或超过奈奎斯特频率时抛出。
    """
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if cutoff_freq <= 0:
        raise ValueError("cutoff_freq must be > 0")
    if order <= 0:
        raise ValueError("order must be > 0")
    nyquist = 0.5 * sr
    if cutoff_freq >= nyquist:
        raise ValueError("cutoff_freq must be less than Nyquist frequency")
    normal_cutoff = cutoff_freq / nyquist
    b, a = sci_signal.butter(order, normal_cutoff, btype="high", analog=False)
    return sci_signal.filtfilt(b, a, signal)


def band_pass_filter(signal, sr, low_cutoff, high_cutoff, order=4):
    """应用巴特沃斯带通滤波。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    low_cutoff : float
        低截止频率（Hz）。
    high_cutoff : float
        高截止频率（Hz）。
    order : int, optional
        滤波器阶数。

    Returns
    -------
    ndarray
        带通滤波后的信号。

    Raises
    ------
    ValueError
        参数非法或超过奈奎斯特频率时抛出。
    """
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if low_cutoff <= 0 or high_cutoff <= 0:
        raise ValueError("cutoff frequencies must be > 0")
    if low_cutoff >= high_cutoff:
        raise ValueError("low_cutoff must be < high_cutoff")
    if order <= 0:
        raise ValueError("order must be > 0")
    nyquist = 0.5 * sr
    if high_cutoff >= nyquist:
        raise ValueError("high_cutoff must be less than Nyquist frequency")
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = sci_signal.butter(order, [low, high], btype="band", analog=False)
    return sci_signal.filtfilt(b, a, signal)


def median_filter(signal, kernel_size=3):
    """应用中值滤波以去除脉冲噪声。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    kernel_size : int, optional
        滤波核大小（必须为奇数）。

    Returns
    -------
    ndarray
        中值滤波后的信号。

    Raises
    ------
    ValueError
        ``kernel_size`` 非正奇数时抛出。
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return sci_signal.medfilt(signal, kernel_size=kernel_size)
