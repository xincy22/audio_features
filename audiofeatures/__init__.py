"""AudioFeatures 包。

提供音频预处理、特征提取、增强与可视化的统一 API，适合快速搭建音频分析流程。

Notes
-----
子模块通过 ``__getattr__`` 延迟导入，以减少启动开销。
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("audio_features")
except PackageNotFoundError:
    __version__ = "0.2.0"

_SUBMODULES = {
    "core",
    "preprocessing",
    "features",
    "augmentation",
    "visualization",
    "pipeline",
    "utils"
}

__all__ = sorted(_SUBMODULES | {"__version__"})


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))
