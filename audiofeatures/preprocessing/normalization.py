"""音频信号归一化工具。"""

import numpy as np
import warnings

from audiofeatures.utils.contract import ensure_float32


def normalize_amplitude(signal, target_dBFS=-20.0):
    """将信号归一化到目标 dBFS。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    target_dBFS : float, optional
        目标 dBFS 值。

    Returns
    -------
    ndarray
        归一化后的信号。

    Raises
    ------
    ValueError
        输入非一维时抛出。

    Warns
    -----
    UserWarning
        当 RMS 接近 0 时返回原信号。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维数组")

    rms = np.sqrt(np.mean(signal**2))
    if np.isclose(rms, 0.0, atol=np.finfo(signal.dtype).eps):
        warnings.warn("信号均方根值接近零，无法进行dBFS归一化，返回原始信号")
        return signal

    current_dBFS = 20 * np.log10(rms)

    gain = 10 ** ((target_dBFS - current_dBFS) / 20.0)
    normalized = signal * gain
    return normalized.astype(np.float32, copy=False)


def peak_normalize(signal, target_peak=0.95):
    """按峰值幅度进行归一化。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    target_peak : float, optional
        目标峰值幅度。

    Returns
    -------
    ndarray
        归一化后的信号。

    Raises
    ------
    ValueError
        输入非一维时抛出。

    Warns
    -----
    UserWarning
        当峰值接近 0 时返回原信号。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维 numpy 数组")

    peak = np.max(np.abs(signal))

    if np.isclose(peak, 0.0, atol=np.finfo(signal.dtype).eps):
        warnings.warn("信号峰值接近零，无法进行峰值归一化，返回原始信号")
        return signal

    normalized = signal * (target_peak / peak)
    return normalized.astype(np.float32, copy=False)


def z_normalize(signal):
    """执行 Z-score 标准化（零均值、单位方差）。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。

    Returns
    -------
    ndarray
        标准化后的信号。

    Raises
    ------
    ValueError
        输入非一维时抛出。

    Warns
    -----
    UserWarning
        当标准差接近 0 时返回原信号。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维 numpy 数组")

    mean = np.mean(signal)
    std = np.std(signal)

    if np.isclose(std, 0.0, atol=np.finfo(signal.dtype).eps):
        warnings.warn("信号标准差接近零，无法进行Z-score标准化，返回原始信号")
        return signal

    normalized = (signal - mean) / std
    return normalized.astype(np.float32, copy=False)


def min_max_normalize(signal, min_val=0.0, max_val=1.0):
    """将信号线性缩放到指定区间。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    min_val : float, optional
        目标最小值。
    max_val : float, optional
        目标最大值。

    Returns
    -------
    ndarray
        缩放后的信号。

    Raises
    ------
    ValueError
        输入非一维或 ``min_val >= max_val`` 时抛出。

    Warns
    -----
    UserWarning
        当信号取值近似常数时返回常数数组。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维 numpy 数组")

    if min_val >= max_val:
        raise ValueError("min_val 必须小于 max_val")

    current_min = np.min(signal)
    current_max = np.max(signal)

    if np.isclose(current_min, current_max, atol=np.finfo(signal.dtype).eps):
        warnings.warn("信号最小值与最大值过于接近，返回填充中间值的数组")
        return np.full_like(signal, (min_val + max_val) / 2, dtype=np.float32)

    normalized = (signal - current_min) / (current_max - current_min)
    normalized = normalized * (max_val - min_val) + min_val
    return normalized.astype(np.float32, copy=False)
