"""Measurement-level error models for the educational simulator.

These functions only edit synthetic pseudorange arrays. They do not generate,
modify, transmit, or decode real GNSS radio-frequency signals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize_satellite_indices(
    num_satellites: int,
    affected_satellites: Sequence[int] | None,
) -> np.ndarray:
    if affected_satellites is None:
        count = max(1, num_satellites // 3)
        return np.arange(count)
    return np.asarray(list(affected_satellites), dtype=int)


def apply_multipath_nlos(
    pseudoranges: np.ndarray,
    bias_level: float = 40.0,
    nlos_probability: float = 0.2,
    rng: np.random.Generator | None = None,
    affected_satellites: Sequence[int] | None = None,
) -> np.ndarray:
    """Add positive range biases representing multipath or NLOS measurements.

    Multipath and non-line-of-sight propagation usually make the path longer
    than the direct geometric path, so this model only adds positive biases.
    """

    if rng is None:
        rng = np.random.default_rng()

    corrupted = np.asarray(pseudoranges, dtype=float).copy()
    if corrupted.ndim != 2:
        raise ValueError("pseudoranges must have shape (num_steps, num_satellites)")

    nlos_probability = float(np.clip(nlos_probability, 0.0, 1.0))
    bias_level = max(float(bias_level), 0.0)
    indices = _normalize_satellite_indices(corrupted.shape[1], affected_satellites)

    event_mask = rng.random((corrupted.shape[0], len(indices))) < nlos_probability
    # Biases vary smoothly enough for plots, but keep random per measurement so
    # students can see residual spikes and geometry-dependent position effects.
    positive_biases = bias_level * (0.5 + rng.random(event_mask.shape))
    corrupted[:, indices] += event_mask * positive_biases

    return corrupted


def apply_spoofing_drift(
    pseudoranges: np.ndarray,
    start_step: int = 40,
    drift_strength: float = 2.0,
    affected_satellites: Sequence[int] | None = None,
) -> np.ndarray:
    """Add a synthetic spoofing-like measurement drift after a selected epoch.

    This is a safe measurement-level anomaly. It is not a signal-generation
    model and cannot be used to transmit or forge real GNSS signals.
    """

    corrupted = np.asarray(pseudoranges, dtype=float).copy()
    if corrupted.ndim != 2:
        raise ValueError("pseudoranges must have shape (num_steps, num_satellites)")

    start_step = int(np.clip(start_step, 0, corrupted.shape[0]))
    drift_strength = float(drift_strength)
    indices = _normalize_satellite_indices(corrupted.shape[1], affected_satellites)

    if start_step >= corrupted.shape[0] or drift_strength == 0.0:
        return corrupted

    drift = np.arange(corrupted.shape[0] - start_step, dtype=float) * drift_strength
    satellite_scales = np.linspace(1.0, 1.5, len(indices))
    corrupted[start_step:, indices] += drift[:, None] * satellite_scales[None, :]

    return corrupted
