"""Simplified 2D DOP/HDOP helpers for educational GNSS geometry analysis."""

from __future__ import annotations

import numpy as np


def build_geometry_matrix(receiver_position: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Build the 2D pseudorange design matrix used for DOP analysis.

    The state is ``[x, y, receiver_clock_bias_m]``. Each row contains the
    line-of-sight unit vector components and the clock-bias derivative. This is
    a simplified 2D analogue of the GNSS geometry matrix.
    """

    receiver_position = np.asarray(receiver_position, dtype=float)
    source_positions = np.asarray(source_positions, dtype=float)

    if receiver_position.shape != (2,):
        raise ValueError("receiver_position must have shape (2,)")
    if source_positions.ndim != 2 or source_positions.shape[1] != 2:
        raise ValueError("source_positions must have shape (num_sources, 2)")
    if len(source_positions) < 3:
        raise ValueError("At least three 2D ranging sources are required for DOP")

    deltas = receiver_position - source_positions
    ranges = np.linalg.norm(deltas, axis=1)
    ranges = np.maximum(ranges, 1e-9)
    return np.column_stack((deltas[:, 0] / ranges, deltas[:, 1] / ranges, np.ones(len(source_positions))))


def compute_dop_matrix(
    receiver_position: np.ndarray,
    source_positions: np.ndarray,
    measurement_sigmas: np.ndarray | None = None,
) -> np.ndarray:
    """Return the covariance-like DOP matrix for the 2D positioning state.

    With equal measurement quality this is ``inv(G.T @ G)``. When measurement
    standard deviations are supplied, this becomes a weighted geometry matrix
    using weights proportional to ``1 / sigma^2``. ``pinv`` is used so poor
    geometries return large finite values rather than crashing.
    """

    geometry = build_geometry_matrix(receiver_position, source_positions)
    if measurement_sigmas is None:
        normal_matrix = geometry.T @ geometry
    else:
        sigmas = np.asarray(measurement_sigmas, dtype=float)
        if sigmas.shape != (len(source_positions),):
            raise ValueError("measurement_sigmas must match the number of sources")
        safe_sigmas = np.maximum(sigmas, 1e-9)
        weights = 1.0 / (safe_sigmas**2)
        normal_matrix = geometry.T @ (weights[:, None] * geometry)

    return np.linalg.pinv(normal_matrix)


def compute_hdop(
    receiver_position: np.ndarray,
    source_positions: np.ndarray,
    measurement_sigmas: np.ndarray | None = None,
) -> float:
    """Compute horizontal DOP for the simplified 2D state."""

    dop_matrix = compute_dop_matrix(receiver_position, source_positions, measurement_sigmas)
    value = np.sqrt(max(float(dop_matrix[0, 0] + dop_matrix[1, 1]), 0.0))
    return float(value)


def compute_gdop(
    receiver_position: np.ndarray,
    source_positions: np.ndarray,
    measurement_sigmas: np.ndarray | None = None,
) -> float:
    """Compute geometry DOP for ``[x, y, clock]`` in the simplified model."""

    dop_matrix = compute_dop_matrix(receiver_position, source_positions, measurement_sigmas)
    value = np.sqrt(max(float(np.trace(dop_matrix)), 0.0))
    return float(value)


def compute_hdop_series(receiver_positions: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Compute HDOP for each receiver epoch.

    ``source_positions`` may be fixed ``(num_sources, 2)`` or time-varying
    ``(num_steps, num_sources, 2)``. This supports GNSS-only and GNSS plus
    LEO-like moving ranging sources.
    """

    receiver_positions = np.asarray(receiver_positions, dtype=float)
    source_positions = np.asarray(source_positions, dtype=float)

    if receiver_positions.ndim != 2 or receiver_positions.shape[1] != 2:
        raise ValueError("receiver_positions must have shape (num_steps, 2)")

    values = np.zeros(len(receiver_positions), dtype=float)
    for step, receiver in enumerate(receiver_positions):
        if source_positions.ndim == 2:
            epoch_sources = source_positions
        elif source_positions.ndim == 3:
            epoch_sources = source_positions[step]
        else:
            raise ValueError("source_positions must be 2D or 3D")
        values[step] = compute_hdop(receiver, epoch_sources)
    return values
