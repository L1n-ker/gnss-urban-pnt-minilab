"""LEO-like moving measurement sources for positioning aid."""

from __future__ import annotations

import numpy as np


def generate_leo_positions(
    num_steps: int = 80,
    num_leo: int = 2,
    radius: float = 1_200_000.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate simple 2D moving LEO-like satellite coordinates.

    The radius is much smaller than the GNSS radius so the moving geometry is
    visible on educational plots. These are abstract ranging beacons, not orbit
    predictions.
    """

    num_steps = int(num_steps)
    num_leo = int(max(num_leo, 0))
    rng = np.random.default_rng(seed)

    positions = np.zeros((num_steps, num_leo, 2), dtype=float)
    if num_steps <= 0 or num_leo == 0:
        return positions

    times = np.arange(num_steps, dtype=float)
    phase_offset = rng.uniform(0.0, 2.0 * np.pi)

    for sat in range(num_leo):
        phase = phase_offset + sat * (2.0 * np.pi / max(num_leo, 1))
        angular_rate = 0.045 + 0.01 * sat
        angles = phase + angular_rate * times
        # Offset the constellation so LEO-like aids add non-redundant geometry.
        center_offset = np.array([25_000.0 * (sat + 1), -15_000.0 * sat])
        positions[:, sat, 0] = center_offset[0] + radius * np.cos(angles)
        positions[:, sat, 1] = center_offset[1] + radius * np.sin(angles)

    return positions


def flatten_time_varying_sources(
    fixed_sources: np.ndarray,
    moving_sources: np.ndarray | None,
) -> np.ndarray:
    """Combine fixed and moving source positions into one time-varying array."""

    fixed_sources = np.asarray(fixed_sources, dtype=float)
    if fixed_sources.ndim != 2 or fixed_sources.shape[1] != 2:
        raise ValueError("fixed_sources must have shape (num_sources, 2)")

    if moving_sources is None:
        return np.repeat(fixed_sources[None, :, :], 1, axis=0)

    moving_sources = np.asarray(moving_sources, dtype=float)
    if moving_sources.ndim != 3 or moving_sources.shape[2] != 2:
        raise ValueError("moving_sources must have shape (num_steps, num_sources, 2)")

    repeated_fixed = np.repeat(fixed_sources[None, :, :], moving_sources.shape[0], axis=0)
    return np.concatenate([repeated_fixed, moving_sources], axis=1)
