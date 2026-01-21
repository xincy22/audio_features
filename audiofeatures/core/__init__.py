"""core 子模块：音频加载与基础信号处理。

包含音频文件读取、元数据获取、分帧与窗函数等基础能力。
"""

from . import audio_loader
from . import signal_processing

from .audio_loader import load_audio, get_audio_info
from .signal_processing import frame_signal, apply_window

__all__ = [
    "audio_loader",
    "signal_processing",
    "load_audio",
    "get_audio_info",
    "frame_signal",
    "apply_window"
]
