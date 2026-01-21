"""特征契约与数组规范化工具。"""

import numpy as np


def ensure_float32(signal, clip=False):
    """确保输入转换为 float32。

    Parameters
    ----------
    signal : array-like
        输入信号。
    clip : bool, optional
        是否裁剪到 [-1, 1]。

    Returns
    -------
    ndarray
        float32 数组。
    """
    arr = np.asarray(signal, dtype=np.float32)
    if clip:
        np.clip(arr, -1.0, 1.0, out=arr)
    return arr


def to_feature_matrix(values, frame_axis=0):
    """将特征转换为 (n_frames, n_features)。

    Parameters
    ----------
    values : array-like
        输入特征。
    frame_axis : int, optional
        输入中的帧轴位置。0 表示输入已是 (n_frames, n_features)，
        1 表示输入为 (n_features, n_frames)。

    Returns
    -------
    ndarray
        形状为 (n_frames, n_features) 的 float32 数组。

    Raises
    ------
    ValueError
        输入维度非法或 frame_axis 不支持时抛出。
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("features must be a 1D or 2D array")
    if frame_axis == 1:
        return arr.T
    if frame_axis != 0:
        raise ValueError("frame_axis must be 0 or 1")
    return arr
