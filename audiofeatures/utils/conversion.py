"""音频相关单位换算工具。"""

import math
import numpy as np


def _to_array(values):
    """将输入转换为 float 数组。"""
    return np.asarray(values, dtype=float)


def _maybe_scalar(values, result):
    """若输入为标量则返回标量结果。"""
    if np.isscalar(values):
        return float(result)
    if isinstance(values, np.ndarray) and values.shape == ():
        return float(result)
    return result


def hz_to_mel(frequencies):
    """将 Hz 频率转换为 Mel 频率。

    Parameters
    ----------
    frequencies : float or ndarray
        输入频率（Hz）。

    Returns
    -------
    float or ndarray
        Mel 频率。
    """
    freqs = _to_array(frequencies)
    mels = 2595.0 * np.log10(1.0 + freqs / 700.0)
    return _maybe_scalar(frequencies, mels)


def mel_to_hz(mels):
    """将 Mel 频率转换为 Hz 频率。

    Parameters
    ----------
    mels : float or ndarray
        Mel 频率。

    Returns
    -------
    float or ndarray
        Hz 频率。
    """
    mels_input = mels
    mels = _to_array(mels)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    return _maybe_scalar(mels_input, freqs)


def hz_to_note(frequency):
    """将频率转换为音符名称。

    Parameters
    ----------
    frequency : float
        频率（Hz）。

    Returns
    -------
    str
        音符名称，例如 ``"A4"``。

    Raises
    ------
    ValueError
        频率非正时抛出。
    """
    if frequency <= 0:
        raise ValueError("frequency must be > 0")

    midi = int(round(69 + 12 * math.log2(frequency / 440.0)))
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_index = midi % 12
    octave = midi // 12 - 1
    return f"{note_names[note_index]}{octave}"


def note_to_hz(note):
    """将音符名称转换为频率（Hz）。

    Parameters
    ----------
    note : str
        音符名称，例如 ``"A4"``、``"C#3"``。

    Returns
    -------
    float
        对应频率（Hz）。

    Raises
    ------
    ValueError
        音符格式非法时抛出。
    """
    if not isinstance(note, str) or len(note) < 2:
        raise ValueError("note must be a string like 'A4' or 'C#3'")

    note = note.strip()
    note_base = note[0].upper()
    accidental = ""
    octave_part = note[1:]
    if note[1:2] in ("#", "b", "B"):
        accidental = note[1:2]
        octave_part = note[2:]

    if accidental == "B":
        accidental = "b"

    try:
        octave = int(octave_part)
    except ValueError as exc:
        raise ValueError("note must include a valid octave number") from exc

    note_names = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11
    }

    key = note_base + accidental
    if key not in note_names:
        raise ValueError("note must be one of A-G with optional # or b")

    midi = (octave + 1) * 12 + note_names[key]
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def seconds_to_samples(seconds, sr):
    """将秒数转换为样本数。

    Parameters
    ----------
    seconds : float
        时间（秒）。
    sr : int
        采样率（Hz）。

    Returns
    -------
    int
        样本数。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    return int(round(seconds * sr))


def samples_to_seconds(samples, sr):
    """将样本数转换为秒数。

    Parameters
    ----------
    samples : int
        样本数。
    sr : int
        采样率（Hz）。

    Returns
    -------
    float
        时间（秒）。

    Raises
    ------
    ValueError
        输入非法时抛出。
    """
    if sr <= 0:
        raise ValueError("sr must be > 0")
    if samples < 0:
        raise ValueError("samples must be >= 0")
    return float(samples) / float(sr)
