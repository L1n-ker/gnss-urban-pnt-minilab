"""Two-agent collaborative positioning toy demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gnss_solver import solve_position_least_squares, solve_time_series
from .simulation import (
    compute_true_ranges,
    generate_pseudoranges,
    generate_receiver_clock_bias,
    generate_receiver_trajectory,
    generate_satellite_positions,
)


@dataclass
class CollaborativeSolution:
    """One-epoch joint positioning result for two agents."""

    positions: np.ndarray
    clock_biases: np.ndarray
    residual_rms: float
    iterations: int
    converged: bool


def _measurement_sigmas(sigmas: np.ndarray | None, count: int) -> np.ndarray:
    if sigmas is None:
        return np.ones(count, dtype=float) * 5.0
    values = np.asarray(sigmas, dtype=float)
    if values.shape != (count,):
        raise ValueError("measurement sigmas must match the number of satellites")
    return np.maximum(values, 1e-6)


def _initial_state(
    pseudoranges_a: np.ndarray,
    pseudoranges_b: np.ndarray,
    satellites: np.ndarray,
    initial_a: np.ndarray | None,
    initial_b: np.ndarray | None,
) -> np.ndarray:
    if initial_a is None:
        solution_a = solve_position_least_squares(pseudoranges_a, satellites)
        state_a = np.array([solution_a.position[0], solution_a.position[1], solution_a.clock_bias], dtype=float)
    else:
        state_a = np.asarray(initial_a, dtype=float)
    if initial_b is None:
        solution_b = solve_position_least_squares(pseudoranges_b, satellites)
        state_b = np.array([solution_b.position[0], solution_b.position[1], solution_b.clock_bias], dtype=float)
    else:
        state_b = np.asarray(initial_b, dtype=float)
    if state_a.shape != (3,) or state_b.shape != (3,):
        raise ValueError("initial states must have shape (3,)")
    return np.concatenate((state_a, state_b))


def _append_agent_rows(
    rows: list[np.ndarray],
    values: list[float],
    state: np.ndarray,
    pseudoranges: np.ndarray,
    satellites: np.ndarray,
    sigmas: np.ndarray,
    offset: int,
) -> None:
    position = state[offset : offset + 2]
    clock_bias = state[offset + 2]
    deltas = position - satellites
    ranges = np.maximum(np.linalg.norm(deltas, axis=1), 1e-9)
    predicted = ranges + clock_bias
    residuals = pseudoranges - predicted

    for index in range(len(satellites)):
        row = np.zeros(6, dtype=float)
        row[offset] = deltas[index, 0] / ranges[index]
        row[offset + 1] = deltas[index, 1] / ranges[index]
        row[offset + 2] = 1.0
        rows.append(row / sigmas[index])
        values.append(float(residuals[index] / sigmas[index]))


def solve_collaborative_epoch(
    pseudoranges_a: np.ndarray,
    pseudoranges_b: np.ndarray,
    satellite_positions: np.ndarray,
    relative_distance_m: float,
    initial_a: np.ndarray | None = None,
    initial_b: np.ndarray | None = None,
    measurement_sigmas_a: np.ndarray | None = None,
    measurement_sigmas_b: np.ndarray | None = None,
    relative_sigma_m: float = 1.0,
    max_iterations: int = 12,
    tolerance: float = 1e-4,
) -> CollaborativeSolution:
    """Solve one epoch with GNSS pseudoranges plus one inter-agent range."""

    pr_a = np.asarray(pseudoranges_a, dtype=float)
    pr_b = np.asarray(pseudoranges_b, dtype=float)
    satellites = np.asarray(satellite_positions, dtype=float)
    if pr_a.shape != pr_b.shape:
        raise ValueError("both agents must use the same number of pseudoranges")
    if satellites.ndim != 2 or satellites.shape[1] != 2 or len(satellites) != len(pr_a):
        raise ValueError("satellite_positions must have shape (num_satellites, 2)")

    sigmas_a = _measurement_sigmas(measurement_sigmas_a, len(pr_a))
    sigmas_b = _measurement_sigmas(measurement_sigmas_b, len(pr_b))
    relative_sigma = max(float(relative_sigma_m), 1e-6)
    state = _initial_state(pr_a, pr_b, satellites, initial_a, initial_b)
    converged = False
    residual_vector = np.zeros(1, dtype=float)

    for iteration in range(1, max_iterations + 1):
        rows: list[np.ndarray] = []
        values: list[float] = []
        _append_agent_rows(rows, values, state, pr_a, satellites, sigmas_a, offset=0)
        _append_agent_rows(rows, values, state, pr_b, satellites, sigmas_b, offset=3)

        delta = state[0:2] - state[3:5]
        distance = max(float(np.linalg.norm(delta)), 1e-9)
        row = np.zeros(6, dtype=float)
        row[0:2] = delta / distance
        row[3:5] = -delta / distance
        rows.append(row / relative_sigma)
        values.append(float((relative_distance_m - distance) / relative_sigma))

        design = np.vstack(rows)
        residual_vector = np.asarray(values, dtype=float)
        update, *_ = np.linalg.lstsq(design, residual_vector, rcond=None)
        state += update
        if np.linalg.norm(update) < tolerance:
            converged = True
            break

    return CollaborativeSolution(
        positions=np.array([[state[0], state[1]], [state[3], state[4]]], dtype=float),
        clock_biases=np.array([state[2], state[5]], dtype=float),
        residual_rms=float(np.sqrt(np.mean(residual_vector**2))),
        iterations=iteration,
        converged=converged,
    )


def collaborative_positioning_results(num_steps: int = 50, seed: int = 41) -> dict[str, np.ndarray | dict[str, float]]:
    """Run a controlled two-agent collaborative-positioning experiment."""

    rng = np.random.default_rng(seed)
    true_a = generate_receiver_trajectory(num_steps=num_steps, seed=seed)
    offset = np.column_stack(
        (
            np.full(num_steps, 78.0),
            26.0 + 5.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, num_steps)),
        )
    )
    true_b = true_a + offset
    satellites = generate_satellite_positions(num_satellites=8, angular_offset=0.35)
    clock_a = generate_receiver_clock_bias(num_steps, seed=seed + 1)
    clock_b = generate_receiver_clock_bias(num_steps, seed=seed + 2) + 8.0

    ranges_a = compute_true_ranges(true_a, satellites)
    ranges_b = compute_true_ranges(true_b, satellites)
    pr_a = generate_pseudoranges(ranges_a, clock_a, noise_std=3.0, rng=rng)
    pr_b = generate_pseudoranges(ranges_b, clock_b, noise_std=3.0, rng=rng)
    sigmas_a = np.ones_like(pr_a) * 4.0
    sigmas_b = np.ones_like(pr_b) * 4.0

    degraded = slice(int(0.35 * num_steps), int(0.72 * num_steps))
    pr_b[degraded, [0, 1, 2]] += np.array([280.0, 230.0, 180.0])
    sigmas_b[degraded, [0, 1, 2]] = 140.0

    gnss_a, clock_est_a, _, _ = solve_time_series(pr_a, satellites, initial_position=true_a[0], method="ols")
    gnss_b, clock_est_b, _, _ = solve_time_series(pr_b, satellites, initial_position=true_b[0], method="ols")

    relative_distances = np.linalg.norm(true_a - true_b, axis=1) + rng.normal(0.0, 0.8, size=num_steps)
    collaborative_a = np.zeros_like(true_a)
    collaborative_b = np.zeros_like(true_b)
    state_a = np.array([gnss_a[0, 0], gnss_a[0, 1], clock_est_a[0]], dtype=float)
    state_b = np.array([gnss_b[0, 0], gnss_b[0, 1], clock_est_b[0]], dtype=float)

    for step in range(num_steps):
        solution = solve_collaborative_epoch(
            pr_a[step],
            pr_b[step],
            satellites,
            relative_distances[step],
            initial_a=state_a,
            initial_b=state_b,
            measurement_sigmas_a=sigmas_a[step],
            measurement_sigmas_b=sigmas_b[step],
            relative_sigma_m=0.9,
        )
        collaborative_a[step] = solution.positions[0]
        collaborative_b[step] = solution.positions[1]
        state_a = np.array([solution.positions[0, 0], solution.positions[0, 1], solution.clock_biases[0]], dtype=float)
        state_b = np.array([solution.positions[1, 0], solution.positions[1, 1], solution.clock_biases[1]], dtype=float)

    gnss_errors_a = np.linalg.norm(gnss_a - true_a, axis=1)
    gnss_errors_b = np.linalg.norm(gnss_b - true_b, axis=1)
    collaborative_errors_a = np.linalg.norm(collaborative_a - true_a, axis=1)
    collaborative_errors_b = np.linalg.norm(collaborative_b - true_b, axis=1)

    metrics = {
        "gnss_only_mean_error_a_m": float(np.mean(gnss_errors_a)),
        "gnss_only_mean_error_b_m": float(np.mean(gnss_errors_b)),
        "collaborative_mean_error_a_m": float(np.mean(collaborative_errors_a)),
        "collaborative_mean_error_b_m": float(np.mean(collaborative_errors_b)),
        "gnss_only_rmse_b_m": float(np.sqrt(np.mean(gnss_errors_b**2))),
        "collaborative_rmse_b_m": float(np.sqrt(np.mean(collaborative_errors_b**2))),
    }
    return {
        "true_positions_a": true_a,
        "true_positions_b": true_b,
        "gnss_positions_a": gnss_a,
        "gnss_positions_b": gnss_b,
        "collaborative_positions_a": collaborative_a,
        "collaborative_positions_b": collaborative_b,
        "gnss_errors_a": gnss_errors_a,
        "gnss_errors_b": gnss_errors_b,
        "collaborative_errors_a": collaborative_errors_a,
        "collaborative_errors_b": collaborative_errors_b,
        "relative_distances": relative_distances,
        "metrics": metrics,
    }
