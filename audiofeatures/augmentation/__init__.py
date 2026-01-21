"""数据增强子模块。"""

from .time_domain import time_stretch, pitch_shift, add_noise, time_mask
from .frequency_domain import (
    spectral_contrast,
    harmonic_enhancement,
    spectral_inversion,
    frequency_mask
)

__all__ = [
    "time_stretch",
    "pitch_shift",
    "add_noise",
    "time_mask",
    "spectral_contrast",
    "harmonic_enhancement",
    "spectral_inversion",
    "frequency_mask"
]
