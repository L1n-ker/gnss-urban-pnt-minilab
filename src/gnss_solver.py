"""Beginner-friendly 2D pseudorange least-squares solver.

The model used here is intentionally small:

    measured_pseudorange = geometric_range + receiver_clock_bias + noise

The unknowns are receiver x, receiver y, and one clock-bias term measured in
meters. Real GNSS is 3D and uses clock bias in seconds, but using meters keeps
the equations easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PositionSolution:
    """Container for one epoch of positioning results."""

    position: np.ndarray
    clock_bias: float
    residual_rms: float
    residuals: np.ndarray
    iterations: int
    converged: bool
    weights: np.ndarray | None = None
    used_mask: np.ndarray | None = None


def _validate_position_inputs(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    initial_guess: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pseudoranges = np.asarray(pseudoranges, dtype=float)
    satellite_positions = np.asarray(satellite_positions, dtype=float)

    if satellite_positions.ndim != 2 or satellite_positions.shape[1] != 2:
        raise ValueError("satellite_positions must have shape (num_satellites, 2)")
    if pseudoranges.ndim != 1:
        raise ValueError("pseudoranges must be a one-dimensional array")
    if len(pseudoranges) != len(satellite_positions):
        raise ValueError("pseudoranges and satellite_positions must have matching lengths")
    if len(pseudoranges) < 3:
        raise ValueError("At least three 2D pseudorange measurements are required")

    if initial_guess is None:
        state = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        state = np.asarray(initial_guess, dtype=float).copy()
        if state.shape != (3,):
            raise ValueError("initial_guess must be shaped (3,)")

    return pseudoranges, satellite_positions, state


def _linearized_design(state: np.ndarray, satellite_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, clock_bias = state
    receiver = np.array([x, y])
    deltas = receiver - satellite_positions
    ranges = np.linalg.norm(deltas, axis=1)
    ranges = np.maximum(ranges, 1e-9)
    predicted = ranges + clock_bias
    design = np.column_stack(
        (
            deltas[:, 0] / ranges,
            deltas[:, 1] / ranges,
            np.ones(len(satellite_positions)),
        )
    )
    return predicted, design


def _solve_weighted_iterative(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    initial_guess: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    max_iterations: int = 12,
    tolerance: float = 1e-4,
) -> PositionSolution:
    pseudoranges, satellite_positions, state = _validate_position_inputs(
        pseudoranges,
        satellite_positions,
        initial_guess,
    )

    if weights is None:
        working_weights = np.ones(len(pseudoranges), dtype=float)
    else:
        working_weights = np.asarray(weights, dtype=float)
        if working_weights.shape != (len(pseudoranges),):
            raise ValueError("weights must match the number of pseudoranges")
        working_weights = np.maximum(working_weights, 1e-12)

    sqrt_weights = np.sqrt(working_weights)
    converged = False
    residuals = np.zeros_like(pseudoranges)

    for iteration in range(1, max_iterations + 1):
        predicted, design = _linearized_design(state, satellite_positions)
        residuals = pseudoranges - predicted
        weighted_design = design * sqrt_weights[:, None]
        weighted_residuals = residuals * sqrt_weights

        update, *_ = np.linalg.lstsq(weighted_design, weighted_residuals, rcond=None)
        state += update

        if np.linalg.norm(update[:2]) < tolerance and abs(update[2]) < tolerance:
            converged = True
            break

    predicted, _ = _linearized_design(state, satellite_positions)
    residuals = pseudoranges - predicted
    residual_rms = float(np.sqrt(np.average(residuals**2, weights=working_weights)))

    return PositionSolution(
        position=state[:2].copy(),
        clock_bias=float(state[2]),
        residual_rms=residual_rms,
        residuals=residuals,
        iterations=iteration,
        converged=converged,
        weights=working_weights.copy(),
    )


def solve_position_least_squares(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    initial_guess: np.ndarray | None = None,
    max_iterations: int = 12,
    tolerance: float = 1e-4,
) -> PositionSolution:
    """Estimate 2D receiver position and clock bias from pseudoranges.

    Parameters
    ----------
    pseudoranges:
        One measurement per satellite, in meters.
    satellite_positions:
        Satellite coordinates shaped ``(num_satellites, 2)``.
    initial_guess:
        Optional ``[x, y, clock_bias_m]`` starting point.

    Returns
    -------
    PositionSolution
        Estimated position, clock bias, residual RMS, and convergence metadata.
    """

    return _solve_weighted_iterative(
        pseudoranges,
        satellite_positions,
        initial_guess=initial_guess,
        weights=None,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def solve_position_weighted_least_squares(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    measurement_sigmas: np.ndarray,
    initial_guess: np.ndarray | None = None,
    max_iterations: int = 12,
    tolerance: float = 1e-4,
) -> PositionSolution:
    """Estimate position with weights proportional to ``1 / sigma^2``."""

    sigmas = np.asarray(measurement_sigmas, dtype=float)
    weights = 1.0 / (np.maximum(sigmas, 1e-9) ** 2)
    return _solve_weighted_iterative(
        pseudoranges,
        satellite_positions,
        initial_guess=initial_guess,
        weights=weights,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _huber_weights(residuals: np.ndarray, huber_delta: float) -> np.ndarray:
    abs_residuals = np.abs(residuals)
    delta = max(float(huber_delta), 1e-9)
    weights = np.ones_like(abs_residuals, dtype=float)
    large = abs_residuals > delta
    weights[large] = delta / np.maximum(abs_residuals[large], 1e-9)
    return weights


def _cauchy_weights(residuals: np.ndarray, cauchy_scale: float) -> np.ndarray:
    scale = max(float(cauchy_scale), 1e-9)
    normalized = residuals / scale
    return 1.0 / (1.0 + normalized**2)


def solve_position_huber(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    initial_guess: np.ndarray | None = None,
    huber_delta: float = 25.0,
    max_outer_iterations: int = 8,
    max_inner_iterations: int = 8,
    tolerance: float = 1e-4,
) -> PositionSolution:
    """Robust pseudorange solution using Huber iterative reweighting."""

    pseudoranges, satellite_positions, state = _validate_position_inputs(
        pseudoranges,
        satellite_positions,
        initial_guess,
    )

    weights = np.ones(len(pseudoranges), dtype=float)
    solution = _solve_weighted_iterative(
        pseudoranges,
        satellite_positions,
        initial_guess=state,
        weights=weights,
        max_iterations=max_inner_iterations,
        tolerance=tolerance,
    )

    for _ in range(max_outer_iterations):
        weights = _huber_weights(solution.residuals, huber_delta)
        next_solution = _solve_weighted_iterative(
            pseudoranges,
            satellite_positions,
            initial_guess=np.array([solution.position[0], solution.position[1], solution.clock_bias]),
            weights=weights,
            max_iterations=max_inner_iterations,
            tolerance=tolerance,
        )
        movement = np.linalg.norm(next_solution.position - solution.position)
        solution = next_solution
        if movement < tolerance:
            break

    solution.weights = weights.copy()
    return solution


def solve_position_cauchy(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    initial_guess: np.ndarray | None = None,
    cauchy_scale: float = 35.0,
    max_outer_iterations: int = 8,
    max_inner_iterations: int = 8,
    tolerance: float = 1e-4,
) -> PositionSolution:
    """Robust pseudorange solution using Cauchy iterative reweighting."""

    pseudoranges, satellite_positions, state = _validate_position_inputs(
        pseudoranges,
        satellite_positions,
        initial_guess,
    )

    weights = np.ones(len(pseudoranges), dtype=float)
    solution = _solve_weighted_iterative(
        pseudoranges,
        satellite_positions,
        initial_guess=state,
        weights=weights,
        max_iterations=max_inner_iterations,
        tolerance=tolerance,
    )

    for _ in range(max_outer_iterations):
        weights = _cauchy_weights(solution.residuals, cauchy_scale)
        next_solution = _solve_weighted_iterative(
            pseudoranges,
            satellite_positions,
            initial_guess=np.array([solution.position[0], solution.position[1], solution.clock_bias]),
            weights=weights,
            max_iterations=max_inner_iterations,
            tolerance=tolerance,
        )
        movement = np.linalg.norm(next_solution.position - solution.position)
        solution = next_solution
        if movement < tolerance:
            break

    solution.weights = weights.copy()
    return solution


def exclude_large_residuals(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    residual_threshold: float = 60.0,
    initial_guess: np.ndarray | None = None,
) -> tuple[PositionSolution, np.ndarray]:
    """Solve once, remove clear residual outliers, and solve again.

    This is a toy measurement-exclusion baseline, not a full RAIM or integrity
    monitoring algorithm.
    """

    initial = solve_position_least_squares(pseudoranges, satellite_positions, initial_guess=initial_guess)
    kept_mask = np.abs(initial.residuals) <= float(residual_threshold)
    if int(np.sum(kept_mask)) < 3:
        kept_mask = np.ones(len(initial.residuals), dtype=bool)

    refined = solve_position_least_squares(
        np.asarray(pseudoranges, dtype=float)[kept_mask],
        np.asarray(satellite_positions, dtype=float)[kept_mask],
        initial_guess=np.array([initial.position[0], initial.position[1], initial.clock_bias]),
    )
    full_residuals = np.asarray(pseudoranges, dtype=float) - (
        np.linalg.norm(np.asarray(satellite_positions, dtype=float) - refined.position, axis=1) + refined.clock_bias
    )
    refined.residuals = full_residuals
    refined.used_mask = kept_mask.copy()
    return refined, kept_mask


def solve_time_series(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    initial_position: np.ndarray | None = None,
    method: str = "ols",
    measurement_sigmas: np.ndarray | None = None,
    huber_delta: float = 25.0,
    cauchy_scale: float = 35.0,
    residual_threshold: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve a sequence of epochs using the previous solution as the next guess.

    ``satellite_positions`` can be either fixed ``(num_satellites, 2)`` geometry
    or time-varying ``(num_steps, num_satellites, 2)`` geometry.
    """

    pseudoranges = np.asarray(pseudoranges, dtype=float)
    satellite_positions = np.asarray(satellite_positions, dtype=float)

    if pseudoranges.ndim != 2:
        raise ValueError("pseudoranges must have shape (num_steps, num_measurements)")

    num_steps = pseudoranges.shape[0]
    positions = np.zeros((num_steps, 2), dtype=float)
    clock_biases = np.zeros(num_steps, dtype=float)
    residual_rms = np.zeros(num_steps, dtype=float)
    converged = np.zeros(num_steps, dtype=bool)

    if initial_position is None:
        guess = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        initial_position = np.asarray(initial_position, dtype=float)
        guess = np.array([initial_position[0], initial_position[1], 0.0], dtype=float)

    for step in range(num_steps):
        if satellite_positions.ndim == 2:
            epoch_satellites = satellite_positions
        elif satellite_positions.ndim == 3:
            epoch_satellites = satellite_positions[step]
        else:
            raise ValueError("satellite_positions must be 2D or 3D")

        epoch_sigmas = None
        if measurement_sigmas is not None:
            sigmas = np.asarray(measurement_sigmas, dtype=float)
            epoch_sigmas = sigmas if sigmas.ndim == 1 else sigmas[step]

        if method == "ols":
            solution = solve_position_least_squares(pseudoranges[step], epoch_satellites, initial_guess=guess)
        elif method == "wls":
            if epoch_sigmas is None:
                epoch_sigmas = np.ones(pseudoranges.shape[1], dtype=float)
            solution = solve_position_weighted_least_squares(
                pseudoranges[step],
                epoch_satellites,
                epoch_sigmas,
                initial_guess=guess,
            )
        elif method == "huber":
            solution = solve_position_huber(
                pseudoranges[step],
                epoch_satellites,
                initial_guess=guess,
                huber_delta=huber_delta,
            )
        elif method == "cauchy":
            solution = solve_position_cauchy(
                pseudoranges[step],
                epoch_satellites,
                initial_guess=guess,
                cauchy_scale=cauchy_scale,
            )
        elif method == "residual_exclusion":
            solution, _ = exclude_large_residuals(
                pseudoranges[step],
                epoch_satellites,
                residual_threshold=residual_threshold,
                initial_guess=guess,
            )
        else:
            raise ValueError("method must be 'ols', 'wls', 'huber', 'cauchy', or 'residual_exclusion'")
        positions[step] = solution.position
        clock_biases[step] = solution.clock_bias
        residual_rms[step] = solution.residual_rms
        converged[step] = solution.converged
        guess = np.array([solution.position[0], solution.position[1], solution.clock_bias])

    return positions, clock_biases, residual_rms, converged
