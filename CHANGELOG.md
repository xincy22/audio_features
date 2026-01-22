# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-01-22
- Defined a frame-level contract: float32 inputs/outputs and `(n_frames, n_features)` shapes.
- Added contract helpers (`ensure_float32`, `to_feature_matrix`) and deterministic augmentation (`rng`/`seed`).
- Added core audio loading, metadata inspection, framing, and windowing utilities.
- Added preprocessing utilities for filtering, normalization, and segmentation.
- Added time-domain, frequency-domain, spectral, and statistical feature extraction.
- Added augmentation and visualization helpers.
- Added feature extraction and aggregation pipelines.
- Expanded docs (contract, batch extraction) and tests; CI now covers Python 3.9–3.13.
- Fixed Py3.9 stats dtype casting in skew/kurtosis.
