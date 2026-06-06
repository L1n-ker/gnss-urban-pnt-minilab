"""Sliding-window least-squares toy optimizer.

This module is inspired by factor-graph GNSS/INS formulations, but it keeps the
implementation deliberately small. Each window estimates 2D position and a
clock-bias term per epoch using pseudorange residuals plus simple odometry
smoothness constraints.
"""

from __future__ import annotations

import numpy as np

from .gnss_solver import solve_time_series


def _epoch_sources(satellite_positions: np.ndarray, step: int) -> np.ndarray:
    if satellite_positions.ndim == 2:
        return satellite_positions
    if satellite_positions.ndim == 3:
        return satellite_positions[step]
    raise ValueError("satellite_positions must be 2D or 3D")


def _epoch_sigmas(measurement_sigmas: np.ndarray | None, step: int, count: int) -> np.ndarray:
    if measurement_sigmas is None:
        return np.ones(count, dtype=float) * 5.0
    sigmas = np.asarray(measurement_sigmas, dtype=float)
    epoch = sigmas if sigmas.ndim == 1 else sigmas[step]
    return np.maximum(epoch, 1e-6)


def _linearized_pseudorange_rows(
    state: np.ndarray,
    satellites: np.ndarray,
    pseudoranges: np.ndarray,
    sigmas: np.ndarray,
    local_index: int,
    window_length: int,
) -> tuple[list[np.ndarray], list[float]]:
    position = state[local_index, :2]
    clock_bias = state[local_index, 2]
    deltas = position - satellites
    ranges = np.maximum(np.linalg.norm(deltas, axis=1), 1e-9)
    predicted = ranges + clock_bias
    residuals = pseudoranges - predicted

    rows: list[np.ndarray] = []
    values: list[float] = []
    state_offset = 3 * local_index
    for source_index in range(len(satellites)):
        row = np.zeros(3 * window_length, dtype=float)
        row[state_offset] = deltas[source_index, 0] / ranges[source_index]
        row[state_offset + 1] = deltas[source_index, 1] / ranges[source_index]
        row[state_offset + 2] = 1.0
        sigma = sigmas[source_index]
        rows.append(row / sigma)
        values.append(float(residuals[source_index] / sigma))
    return rows, values


def _append_motion_rows(
    rows: list[np.ndarray],
    values: list[float],
    state: np.ndarray,
    odometry_deltas: np.ndarray,
    start_step: int,
    window_length: int,
    odometry_sigma_m: float,
) -> None:
    sigma = max(float(odometry_sigma_m), 1e-6)
    for local_index in range(1, window_length):
        global_step = start_step + local_index
        observed_delta = odometry_deltas[global_step - 1]
        predicted_delta = state[local_index, :2] - state[local_index - 1, :2]
        residual = observed_delta - predicted_delta
        for axis in range(2):
            row = np.zeros(3 * window_length, dtype=float)
            row[3 * (local_index - 1) + axis] = -1.0
            row[3 * local_index + axis] = 1.0
            rows.append(row / sigma)
            values.append(float(residual[axis] / sigma))


def _append_prior_rows(
    rows: list[np.ndarray],
    values: list[float],
    state: np.ndarray,
    prior: np.ndarray,
    window_length: int,
    prior_sigma_m: float,
) -> None:
    sigma = max(float(prior_sigma_m), 1e-6)
    residual = prior - state[0]
    for component in range(3):
        row = np.zeros(3 * window_length, dtype=float)
        row[component] = 1.0
        rows.append(row / sigma)
        values.append(float(residual[component] / sigma))


def solve_sliding_window_time_series(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    odometry_deltas: np.ndarray,
    measurement_sigmas: np.ndarray | None = None,
    window_size: int = 5,
    initial_position: np.ndarray | None = None,
    odometry_sigma_m: float = 1.2,
    prior_sigma_m: float = 2.0,
    max_iterations: int = 6,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a time series with a simplified sliding-window optimizer."""

    pseudoranges = np.asarray(pseudoranges, dtype=float)
    satellite_positions = np.asarray(satellite_positions, dtype=float)
    odometry_deltas = np.asarray(odometry_deltas, dtype=float)
    if pseudoranges.ndim != 2:
        raise ValueError("pseudoranges must have shape (num_steps, num_measurements)")

    num_steps = pseudoranges.shape[0]
    if odometry_deltas.shape != (num_steps - 1, 2):
        raise ValueError("odometry_deltas must have shape (num_steps - 1, 2)")

    method = "wls" if measurement_sigmas is not None else "ols"
    baseline_positions, baseline_clocks, _, _ = solve_time_series(
        pseudoranges,
        satellite_positions,
        initial_position=initial_position,
        method=method,
        measurement_sigmas=measurement_sigmas,
    )

    positions = np.zeros((num_steps, 2), dtype=float)
    clocks = np.zeros(num_steps, dtype=float)
    window_size = int(max(window_size, 2))

    for end_step in range(num_steps):
        start_step = max(0, end_step - window_size + 1)
        window_indices = np.arange(start_step, end_step + 1)
        window_length = len(window_indices)
        state = np.column_stack(
            (
                baseline_positions[window_indices],
                baseline_clocks[window_indices],
            )
        )

        if start_step == 0:
            if initial_position is None:
                prior = state[0].copy()
            else:
                initial = np.asarray(initial_position, dtype=float)
                prior = np.array([initial[0], initial[1], state[0, 2]], dtype=float)
        else:
            prior = np.array([positions[start_step, 0], positions[start_step, 1], clocks[start_step]], dtype=float)
            state[0] = prior

        for _ in range(max_iterations):
            rows: list[np.ndarray] = []
            values: list[float] = []
            for local_index, global_step in enumerate(window_indices):
                epoch_satellites = _epoch_sources(satellite_positions, int(global_step))
                sigmas = _epoch_sigmas(measurement_sigmas, int(global_step), len(epoch_satellites))
                pr_rows, pr_values = _linearized_pseudorange_rows(
                    state,
                    epoch_satellites,
                    pseudoranges[global_step],
                    sigmas,
                    local_index,
                    window_length,
                )
                rows.extend(pr_rows)
                values.extend(pr_values)

            _append_motion_rows(
                rows,
                values,
                state,
                odometry_deltas,
                start_step,
                window_length,
                odometry_sigma_m,
            )
            _append_prior_rows(rows, values, state, prior, window_length, prior_sigma_m)

            design = np.vstack(rows)
            residual = np.asarray(values, dtype=float)
            update, *_ = np.linalg.lstsq(design, residual, rcond=None)
            state += update.reshape(window_length, 3)
            if np.linalg.norm(update) < tolerance:
                break

        positions[end_step] = state[-1, :2]
        clocks[end_step] = state[-1, 2]

    return positions, clocks
