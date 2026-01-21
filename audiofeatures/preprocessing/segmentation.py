"""音频分段工具。"""

import numpy as np
import warnings

from audiofeatures.utils.contract import ensure_float32


def segment_by_energy(signal, sr, threshold=0.05, min_length=0.1):
    """基于能量阈值进行分段。

    Parameters
    ----------
    signal : ndarray
        一维输入信号（建议已归一化到 [-1, 1]）。
    sr : int
        采样率（Hz）。
    threshold : float, optional
        能量阈值，范围 [0, 1]。
    min_length : float, optional
        片段最小时长（秒）。

    Returns
    -------
    list of tuple
        片段列表，每个元素为 ``(start_idx, end_idx)``。

    Raises
    ------
    ValueError
        输入非法或参数越界时抛出。

    Warns
    -----
    UserWarning
        当信号能量接近 0 时返回空列表。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维 numpy 数组")
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError("采样率必须是正整数")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if min_length < 0:
        raise ValueError("min_length must be >= 0")

    energy = signal ** 2
    max_energy = np.max(energy)

    if np.isclose(max_energy, 0.0, atol=np.finfo(signal.dtype).eps):
        warnings.warn("信号能量过低，可能无法检测到有效片段")
        return []

    norm_energy = energy / max_energy

    indices = np.where(norm_energy > threshold)[0]
    if len(indices) == 0:
        return []

    segments = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > 1:
            end = indices[i - 1]
            if (end - start) / sr >= min_length:
                segments.append((start, end))
            start = indices[i]

    end = indices[-1]
    if (end - start) / sr >= min_length:
        segments.append((start, end))

    return segments


def segment_by_zcr(
        signal, sr,
        threshold=0.2, min_length=0.1,
        frame_length=0.025, hop_length=0.010
    ):
    """基于过零率（ZCR）进行分段。

    Parameters
    ----------
    signal : ndarray
        一维输入信号（建议已归一化到 [-1, 1]）。
    sr : int
        采样率（Hz）。
    threshold : float, optional
        ZCR 阈值，范围 [0, 1]。
    min_length : float, optional
        片段最小时长（秒）。
    frame_length : float, optional
        分析帧长度（秒）。
    hop_length : float, optional
        帧移（秒）。

    Returns
    -------
    list of tuple
        片段列表，每个元素为 ``(start_idx, end_idx)``。

    Raises
    ------
    ValueError
        输入非法或参数越界时抛出。

    Warns
    -----
    UserWarning
        当信号长度不足一帧时返回空列表。
    """
    signal = ensure_float32(signal)
    if signal.ndim != 1:
        raise ValueError("输入信号必须是一维 numpy 数组")
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError("采样率必须是正整数")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if min_length < 0:
        raise ValueError("min_length must be >= 0")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be > 0")
    if frame_length < hop_length:
        raise ValueError("frame_length must be >= hop_length")

    frame_length_samples = int(frame_length * sr)
    hop_length_samples = int(hop_length * sr)

    if len(signal) < frame_length_samples:
        warnings.warn("信号长度小于帧长度，无法进行分段")
        return []

    zcr = []
    for i in range(0, len(signal) - frame_length_samples, hop_length_samples):
        frame = signal[i:i + frame_length_samples]
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame).astype(int))))
        zcr.append(zero_crossings / (frame_length_samples - 1))

    zcr = np.array(zcr)
    active = zcr > threshold

    segments = []
    in_segment = False
    start_frame = 0

    for i, is_active in enumerate(active):
        if is_active and not in_segment:
            in_segment = True
            start_frame = i
        elif not is_active and in_segment:
            in_segment = False
            start_idx = start_frame * hop_length_samples
            end_idx = i * hop_length_samples + frame_length_samples
            if (end_idx - start_idx) / sr >= min_length:
                segments.append((start_idx, end_idx))
    if in_segment:
        start_idx = start_frame * hop_length_samples
        end_idx = len(active) * hop_length_samples + frame_length_samples
        if (end_idx - start_idx) / sr >= min_length:
            segments.append((start_idx, end_idx))
    return segments
