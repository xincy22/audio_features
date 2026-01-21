"""音频加载与元信息读取。

提供统一的音频读取接口以及基础元数据提取能力，解码依赖 ``librosa``，
无损格式元信息优先使用 ``soundfile``。
"""

import librosa
import numpy as np
import os
import soundfile as sf


def load_audio(file_path, sr=None, mono=True, offset=0.0, duration=None):
    """加载音频文件。

    Parameters
    ----------
    file_path : str
        音频文件路径。
    sr : int or None, optional
        目标采样率。为 ``None`` 时保持原采样率。
    mono : bool, optional
        是否转换为单声道。
    offset : float, optional
        起始读取时间（秒）。
    duration : float or None, optional
        读取时长（秒）。为 ``None`` 时读取全部。

    Returns
    -------
    tuple
        ``(audio, sample_rate)``，其中 ``audio`` 为一维 ``ndarray``。

    Raises
    ------
    Exception
        读取失败时抛出异常。

    Notes
    -----
    - MP3 等有损格式的解码依赖系统后端（如 ffmpeg）。
    - 大文件建议使用 ``offset`` 与 ``duration`` 控制内存占用。

    Examples
    --------
    >>> from audiofeatures.core import load_audio
    >>> audio, sr = load_audio("example.wav", sr=16000)
    >>> audio.shape, sr
    ((16000,), 16000)
    """
    try:
        audio, sample_rate = librosa.load(
            path=file_path,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration
        )
        return audio, sample_rate
    except Exception as e:
        raise Exception(f"加载音频文件失败 '{file_path}': {str(e)}")


def get_audio_info(file_path):
    """获取音频文件的基础元数据。

    Parameters
    ----------
    file_path : str
        音频文件路径。

    Returns
    -------
    dict
        元信息字典，包含 ``sr``、``channels``、``duration``、``samples``、
        ``bit_depth`` 与 ``format``。

    Raises
    ------
    Exception
        获取失败时抛出异常。

    Notes
    -----
    - WAV/FLAC/OGG/AIFF 等无损格式使用 ``soundfile`` 读取。
    - MP3 等格式通过解码后的样本估算时长与通道数。
    """
    try:
        file_format = os.path.splitext(file_path)[1].lower().replace(".", "")

        info = {}

        if file_format in ["wav", "flac", "ogg", "aiff"]:
            sf_info = sf.info(file_path)
            info["sr"] = sf_info.samplerate
            info["channels"] = sf_info.channels
            info["duration"] = sf_info.duration
            info["samples"] = sf_info.frames
            info["format"] = file_format

            if hasattr(sf_info, "subtype"):
                if "16" in sf_info.subtype:
                    info["bit_depth"] = 16
                elif "24" in sf_info.subtype:
                    info["bit_depth"] = 24
                elif "32" in sf_info.subtype:
                    info["bit_depth"] = 32
                elif "8" in sf_info.subtype:
                    info["bit_depth"] = 8
                else:
                    info["bit_depth"] = None
            else:
                info["bit_depth"] = None
        else:
            y, sr = librosa.load(file_path, sr=None, mono=False)

            if y.ndim > 1:
                info["channels"] = y.shape[0]
            else:
                info["channels"] = 1

            info["sr"] = sr
            info["duration"] = librosa.get_duration(y=y, sr=sr)
            info["samples"] = y.shape[-1]
            info["format"] = file_format
            info["bit_depth"] = None

        return info
    except Exception as e:
        raise Exception(f"获取音频信息失败 '{file_path}': {str(e)}")
