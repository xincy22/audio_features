"""信号分帧与窗函数处理。"""

import numpy as np
from scipy import signal as scipy_signal


def frame_signal(signal, frame_length, hop_length, center=True):
    """将一维信号切分为重叠帧。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    frame_length : int
        帧长度（样本数）。
    hop_length : int
        帧移（样本数）。
    center : bool, optional
        是否居中对齐。为 ``True`` 时在两端补零。

    Returns
    -------
    ndarray
        分帧结果，形状为 ``(n_frames, frame_length)``。

    Raises
    ------
    ValueError
        输入非一维或参数非法时抛出。

    Notes
    -----
    当信号长度不足一个帧时返回空数组。
    """
    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维数组")
    if frame_length <= 0:
        raise ValueError("frame_length must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    if center:
        pad_length = frame_length // 2
        signal = np.pad(signal, pad_length, mode="constant")
    if len(signal) < frame_length:
        return np.zeros((0, frame_length))
    num_frames = (len(signal) - frame_length) // hop_length + 1
    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * hop_length
        frames[i] = signal[start:start + frame_length]
    return frames


def apply_window(frames, window_type="hann"):
    """对分帧信号应用窗函数。

    Parameters
    ----------
    frames : ndarray
        二维分帧数组，形状 ``(n_frames, frame_length)``。
    window_type : str, optional
        窗函数类型，支持 ``hann``、``hamming``、``blackman``、
        ``bartlett``、``kaiser``、``rectangular``。

    Returns
    -------
    ndarray
        加窗后的帧，形状与输入一致。

    Raises
    ------
    ValueError
        输入维度或窗函数类型非法时抛出。

    Notes
    -----
    加窗会改变幅值分布，必要时可做能量补偿。
    """
    if frames.ndim != 2:
        raise ValueError("输入帧必须是二维数组")

    frame_length = frames.shape[1]

    if window_type == "rectangular":
        window = np.ones(frame_length)
    elif window_type == "kaiser":
        window = scipy_signal.windows.kaiser(frame_length, beta=14)
    else:
        try:
            window_func = getattr(scipy_signal.windows, window_type)
            window = window_func(frame_length)
        except AttributeError:
            supported_windows = [
                "hann", "hamming", "blackman",
                "bartlett", "kaiser", "rectangular"
            ]
            raise ValueError(
                f"Unsupported window type '{window_type}'. "
                f"Supported types: {supported_windows}"
            )
    return frames * window
