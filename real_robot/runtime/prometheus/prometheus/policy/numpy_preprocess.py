"""Prometheus numpy-window adapter; tactile logic lives in flow_matching.infer.preprocess."""

from __future__ import annotations

from infer.preprocess import build_obs_from_numpy_frames, timestamp_dict

__all__ = ("build_obs_from_numpy_frames", "timestamp_dict")
