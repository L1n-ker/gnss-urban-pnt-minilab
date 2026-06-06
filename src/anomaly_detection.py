"""Simple residual-based anomaly detection helpers."""

from __future__ import annotations

import numpy as np


def detect_residual_anomalies(residual_rms: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    """Flag epochs whose residual RMS exceeds a fixed meter-level threshold."""

    residual_rms = np.asarray(residual_rms, dtype=float)
    return residual_rms > float(threshold)


def robust_residual_threshold(residual_rms: np.ndarray, scale: float = 4.0) -> float:
    """Return a robust threshold using median absolute deviation.

    This is optional in the UI, but useful for students: the threshold adapts to
    the typical residual level without needing a complicated detector.
    """

    residual_rms = np.asarray(residual_rms, dtype=float)
    if residual_rms.size == 0:
        return 0.0

    median = float(np.median(residual_rms))
    mad = float(np.median(np.abs(residual_rms - median)))
    sigma_like = 1.4826 * mad
    return median + scale * sigma_like
