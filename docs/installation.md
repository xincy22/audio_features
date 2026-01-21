# 安装与环境

本项目适合本地开发使用，也支持构建 wheel 在其它项目中安装。

## 环境要求

- Python 3.9 - 3.13
- 依赖: `numpy`, `scipy`, `librosa`, `soundfile`, `numba`
- 可视化（可选）: `matplotlib`

如果安装 `soundfile` 时报错，可能需要先安装系统的 `libsndfile`（Linux: `apt-get install libsndfile1`，
macOS: `brew install libsndfile`）。

## 从 PyPI 安装

```bash
pip install audio-features
# 或
pip install audio_features
```

## 本地开发（推荐）

在其它项目的虚拟环境里执行：

```bash
pip install -e path\to\audio_features
# 或在仓库根目录：
pip install -e .
```

这样修改 `audiofeatures` 的代码会立即生效。

## 从源码安装

```bash
pip install .
```

## 生成并安装 wheel

```bash
python -m build
pip install dist\audio_features-0.2.0-py3-none-any.whl
```

## 可选依赖

```bash
pip install "audio-features[viz]"
pip install "audio-features[docs]"
```

本地源码场景可用 `pip install -e ".[viz]"` 或 `pip install -e ".[docs]"`。

## MP3 相关说明

`librosa` 读取 MP3 依赖系统后端（如 ffmpeg）。若 MP3 加载失败，请安装 ffmpeg 或使用
WAV/FLAC 等格式。
