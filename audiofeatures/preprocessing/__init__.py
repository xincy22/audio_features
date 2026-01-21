"""预处理子模块。

包含滤波、归一化与分段等常用预处理工具。
"""

from . import filtering
from . import normalization
from . import segmentation

from .filtering import (
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    median_filter
)
from .normalization import (
    normalize_amplitude,
    peak_normalize,
    z_normalize,
    min_max_normalize
)
from .segmentation import (
    segment_by_energy,
    segment_by_zcr
)

__all__ = [
    "filtering",
    "normalization",
    "segmentation",
    "low_pass_filter",
    "high_pass_filter",
    "band_pass_filter",
    "median_filter",
    "normalize_amplitude",
    "peak_normalize",
    "z_normalize",
    "min_max_normalize",
    "segment_by_energy",
    "segment_by_zcr"
]
