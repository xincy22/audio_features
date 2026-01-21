"""音频与特征文件 I/O 工具。"""

import numpy as np
import soundfile as sf

from audiofeatures.core.audio_loader import load_audio as core_load_audio


def load_audio(file_path, sr=None, mono=True):
    """读取音频文件。

    Parameters
    ----------
    file_path : str
        音频文件路径。
    sr : int or None, optional
        目标采样率，为 ``None`` 时保持原采样率。
    mono : bool, optional
        是否转换为单声道。

    Returns
    -------
    tuple
        ``(signal, sr)``。

    Notes
    -----
    实际读取由 ``audiofeatures.core.audio_loader.load_audio`` 完成。
    """
    return core_load_audio(file_path, sr=sr, mono=mono)


def save_audio(signal, sr, file_path):
    """保存音频到文件。

    Parameters
    ----------
    signal : ndarray
        一维输入信号。
    sr : int
        采样率（Hz）。
    file_path : str
        输出文件路径。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if sr <= 0:
        raise ValueError("sr must be > 0")

    sf.write(file_path, signal, sr)


def save_features(features, file_path):
    """将特征字典保存为 ``.npz`` 文件。

    Parameters
    ----------
    features : dict
        特征字典，键为名称、值为数组。
    file_path : str
        输出路径（建议以 ``.npz`` 结尾）。

    Raises
    ------
    ValueError
        ``features`` 不是字典时抛出。
    """
    if not isinstance(features, dict):
        raise ValueError("features must be a dict")
    arrays = {key: np.asarray(value) for key, value in features.items()}
    np.savez_compressed(file_path, **arrays)


def load_features(file_path):
    """从 ``.npz`` 文件读取特征。

    Parameters
    ----------
    file_path : str
        特征文件路径。

    Returns
    -------
    dict
        特征字典。
    """
    with np.load(file_path) as data:
        return {key: data[key] for key in data.files}
