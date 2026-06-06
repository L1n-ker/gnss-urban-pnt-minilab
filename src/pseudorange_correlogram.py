"""Simplified pseudorange-correlogram positioning utilities.

This module is an educational, 2D synthetic-measurement reproduction of one
idea from Vicenzo et al. (2024): score candidate receiver states by comparing
candidate ranges with incoming pseudoranges, then weight the per-satellite
score by C/N0. It does not parse UrbanNav RINEX pseudoranges or real C/N0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


SPEED_OF_LIGHT_MPS = 299_792_458.0
GPS_L1_CA_CHIP_RATE_HZ = 1.023e6
DEFAULT_CHIP_LENGTH_M = SPEED_OF_LIGHT_MPS / GPS_L1_CA_CHIP_RATE_HZ
EARTH_RADIUS_M = 6_371_000.0
URBANNAV_TST_GT_URL = (
    "https://www.dropbox.com/scl/fi/9osse97q6wn2brnz44gjl/"
    "UrbanNav_TST_GT_raw.txt?rlkey=wb3x0g86hpjpjus9uggh3jtl8&dl=1"
)


@dataclass
class CorrelogramSolution:
    """Result of one simplified pseudorange-correlogram grid search."""

    position: np.ndarray
    clock_bias: float
    score: float
    score_grid: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    clock_bias_grid: np.ndarray | None = None
    full_score_grid: np.ndarray | None = None


def _dms_to_decimal(degrees: str, minutes: str, seconds: str) -> float:
    deg = float(degrees)
    sign = -1.0 if deg < 0.0 else 1.0
    return sign * (abs(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0)


def parse_urbannav_ground_truth_text(text: str, max_points: int | None = None) -> np.ndarray:
    """Parse UrbanNav ground-truth text into local East-North meter positions."""

    latitudes: list[float] = []
    longitudes: list[float] = []

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 19:
            continue
        try:
            float(parts[0])
            latitude = _dms_to_decimal(parts[3], parts[4], parts[5])
            longitude = _dms_to_decimal(parts[6], parts[7], parts[8])
        except ValueError:
            continue

        latitudes.append(latitude)
        longitudes.append(longitude)
        if max_points is not None and len(latitudes) >= int(max_points):
            break

    if not latitudes:
        raise ValueError("No UrbanNav ground-truth rows were parsed")

    lat = np.radians(np.asarray(latitudes, dtype=float))
    lon = np.radians(np.asarray(longitudes, dtype=float))
    lat0 = lat[0]
    lon0 = lon[0]

    east = (lon - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    north = (lat - lat0) * EARTH_RADIUS_M
    return np.column_stack((east, north))


def load_urbannav_ground_truth_file(path: Path | str, max_points: int | None = None) -> np.ndarray:
    """Load a local UrbanNav ground-truth text file."""

    text = Path(path).read_text(encoding="utf-8")
    return parse_urbannav_ground_truth_text(text, max_points=max_points)


def ensure_urbannav_ground_truth_file(path: Path | str) -> Path:
    """Download the small public UrbanNav TST ground-truth file if missing."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(URBANNAV_TST_GT_URL, destination)
    return destination


def _normalized_cn0_weights(cn0_dbhz: np.ndarray) -> np.ndarray:
    cn0 = np.asarray(cn0_dbhz, dtype=float)
    if cn0.ndim != 1:
        raise ValueError("cn0_dbhz must be a one-dimensional array")
    max_cn0 = float(np.max(cn0))
    if max_cn0 <= 0.0:
        return np.ones_like(cn0)
    return np.clip(cn0 / max_cn0, 0.0, 1.0)


def pseudorange_correlogram_score(
    candidate_position: np.ndarray,
    satellite_positions: np.ndarray,
    pseudoranges: np.ndarray,
    cn0_dbhz: np.ndarray,
    clock_bias_m: float,
    chip_length_m: float = DEFAULT_CHIP_LENGTH_M,
) -> float:
    """Score one candidate using triangular pseudorange consistency.

    The score follows the spirit of the paper's pseudorange-correlation term:
    a candidate receives high score when candidate range + clock bias is close
    to the measured pseudorange, and zero score when the discrepancy exceeds
    one code chip length. C/N0 is used as a normalized satellite weight.
    """

    candidate = np.asarray(candidate_position, dtype=float)
    satellites = np.asarray(satellite_positions, dtype=float)
    ranges = np.linalg.norm(satellites - candidate, axis=1)
    pseudoranges = np.asarray(pseudoranges, dtype=float)

    if candidate.shape != (2,):
        raise ValueError("candidate_position must have shape (2,)")
    if satellites.ndim != 2 or satellites.shape[1] != 2:
        raise ValueError("satellite_positions must have shape (num_satellites, 2)")
    if pseudoranges.shape != (len(satellites),):
        raise ValueError("pseudoranges must match the number of satellites")

    weights = _normalized_cn0_weights(cn0_dbhz)
    if weights.shape != pseudoranges.shape:
        raise ValueError("cn0_dbhz must match the number of pseudoranges")

    chip_length = max(float(chip_length_m), 1e-9)
    residual_m = np.abs((ranges + float(clock_bias_m)) - pseudoranges)
    correlations = np.maximum(0.0, 1.0 - residual_m / chip_length)
    return float(np.sum(correlations * weights) / np.sum(np.maximum(weights, 1e-12)))


def solve_position_pseudorange_correlogram(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    cn0_dbhz: np.ndarray,
    initial_position: np.ndarray,
    grid_radius_m: float = 180.0,
    grid_step_m: float = 12.0,
    chip_length_m: float = DEFAULT_CHIP_LENGTH_M,
    clock_bias_search_m: np.ndarray | None = None,
) -> CorrelogramSolution:
    """Solve one epoch by grid-searching the highest correlogram score.

    If ``clock_bias_search_m`` is supplied, the candidate state is
    ``[x, y, clock_bias_m]`` and the returned ``score_grid`` is the XY slice at
    the best clock-bias value. Clock bias is expressed in meters, matching the
    rest of this MiniLab's simplified pseudorange model. If it is omitted, the
    solver keeps the older compact behavior and estimates one median residual
    clock-bias value for each XY candidate.
    """

    pseudoranges = np.asarray(pseudoranges, dtype=float)
    satellites = np.asarray(satellite_positions, dtype=float)
    center = np.asarray(initial_position, dtype=float)

    if center.shape != (2,):
        raise ValueError("initial_position must have shape (2,)")
    if len(pseudoranges) < 3:
        raise ValueError("At least three pseudoranges are required")

    radius = max(float(grid_radius_m), 0.0)
    step = max(float(grid_step_m), 1e-6)
    x_grid = np.arange(center[0] - radius, center[0] + radius + 0.5 * step, step)
    y_grid = np.arange(center[1] - radius, center[1] + radius + 0.5 * step, step)

    best_score = -np.inf
    best_position = center.copy()
    best_clock_bias = 0.0
    best_clock_index = 0

    if clock_bias_search_m is None:
        score_grid = np.full((len(y_grid), len(x_grid)), np.nan, dtype=float)
        full_score_grid = None
        clock_bias_grid = None

        for y_index, y_value in enumerate(y_grid):
            for x_index, x_value in enumerate(x_grid):
                candidate = np.array([x_value, y_value], dtype=float)
                ranges = np.linalg.norm(satellites - candidate, axis=1)
                clock_bias = float(np.median(pseudoranges - ranges))
                score = pseudorange_correlogram_score(
                    candidate,
                    satellites,
                    pseudoranges,
                    cn0_dbhz,
                    clock_bias,
                    chip_length_m=chip_length_m,
                )
                score_grid[y_index, x_index] = score
                if score > best_score:
                    best_score = score
                    best_position = candidate
                    best_clock_bias = clock_bias
    else:
        clock_bias_grid = np.asarray(clock_bias_search_m, dtype=float)
        if clock_bias_grid.ndim != 1 or len(clock_bias_grid) == 0:
            raise ValueError("clock_bias_search_m must be a non-empty one-dimensional array")

        full_score_grid = np.full((len(clock_bias_grid), len(y_grid), len(x_grid)), np.nan, dtype=float)
        for clock_index, clock_bias in enumerate(clock_bias_grid):
            for y_index, y_value in enumerate(y_grid):
                for x_index, x_value in enumerate(x_grid):
                    candidate = np.array([x_value, y_value], dtype=float)
                    score = pseudorange_correlogram_score(
                        candidate,
                        satellites,
                        pseudoranges,
                        cn0_dbhz,
                        float(clock_bias),
                        chip_length_m=chip_length_m,
                    )
                    full_score_grid[clock_index, y_index, x_index] = score
                    if score > best_score:
                        best_score = score
                        best_position = candidate
                        best_clock_bias = float(clock_bias)
                        best_clock_index = clock_index
        score_grid = full_score_grid[best_clock_index].copy()

    return CorrelogramSolution(
        position=best_position,
        clock_bias=best_clock_bias,
        score=float(best_score),
        score_grid=score_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        clock_bias_grid=clock_bias_grid,
        full_score_grid=full_score_grid,
    )


def generate_synthetic_correlogram_case(
    true_positions: np.ndarray,
    seed: int = 2026,
    num_satellites: int = 8,
    scenario: str = "nlos_urban",
) -> dict[str, np.ndarray]:
    """Generate synthetic pseudorange/C/N0 measurements on a public route.

    Synthetic assumptions:
    - LOS: small zero-mean pseudorange noise and high C/N0.
    - Multipath: moderate positive bias/noise and moderate C/N0.
    - NLOS: larger positive bias/noise and low C/N0.

    The public route gives only the trajectory context here. The pseudoranges
    and C/N0 are generated by this function and are not UrbanNav RINEX values.
    """

    true_positions = np.asarray(true_positions, dtype=float)
    if true_positions.ndim != 2 or true_positions.shape[1] != 2:
        raise ValueError("true_positions must have shape (num_steps, 2)")

    rng = np.random.default_rng(seed)
    angles = 0.15 + np.linspace(0.0, 2.0 * np.pi, num_satellites, endpoint=False)
    radius = 20_200_000.0
    satellite_positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

    num_steps = len(true_positions)
    geometric_ranges = np.linalg.norm(true_positions[:, None, :] - satellite_positions[None, :, :], axis=2)
    clock_bias = 42.0 + 5.0 * np.sin(np.linspace(0.0, 2.5 * np.pi, num_steps))
    clock_bias += np.cumsum(rng.normal(0.0, 0.04, size=num_steps))
    scenario_key = scenario.lower().replace("-", "_")
    if scenario_key not in {"clean_open_sky", "nlos_urban"}:
        raise ValueError("scenario must be 'clean_open_sky' or 'nlos_urban'")

    los_noise_std = 1.8 if scenario_key == "clean_open_sky" else 2.5
    pseudoranges = geometric_ranges + clock_bias[:, None] + rng.normal(0.0, los_noise_std, size=geometric_ranges.shape)
    cn0 = rng.normal(46.0, 0.9, size=geometric_ranges.shape)

    multipath_mask = np.zeros_like(pseudoranges, dtype=bool)
    nlos_mask = np.zeros_like(pseudoranges, dtype=bool)

    if scenario_key == "nlos_urban":
        mp_start = int(0.18 * num_steps)
        mp_stop = int(0.88 * num_steps)
        nlos_start = int(0.32 * num_steps)
        nlos_stop = int(0.82 * num_steps)
        mp_affected = np.array([4, 5])
        nlos_affected = np.array([0, 1, 2, 3])

        multipath_mask[mp_start:mp_stop, mp_affected] = True
        nlos_mask[nlos_start:nlos_stop, nlos_affected] = True

        pseudoranges[multipath_mask] += rng.uniform(45.0, 120.0, size=int(np.sum(multipath_mask)))
        pseudoranges[multipath_mask] += rng.normal(0.0, 10.0, size=int(np.sum(multipath_mask)))
        cn0[multipath_mask] = rng.uniform(29.0, 37.0, size=int(np.sum(multipath_mask)))

        pseudoranges[nlos_mask] += rng.uniform(320.0, 650.0, size=int(np.sum(nlos_mask)))
        pseudoranges[nlos_mask] += rng.normal(0.0, 25.0, size=int(np.sum(nlos_mask)))
        cn0[nlos_mask] = rng.uniform(10.0, 18.0, size=int(np.sum(nlos_mask)))

    return {
        "true_positions": true_positions,
        "satellite_positions": satellite_positions,
        "pseudoranges": pseudoranges,
        "clock_bias": clock_bias,
        "cn0": cn0,
        "multipath_mask": multipath_mask,
        "nlos_mask": nlos_mask,
        "scenario": np.array(scenario_key),
    }


def solve_correlogram_time_series(
    pseudoranges: np.ndarray,
    satellite_positions: np.ndarray,
    cn0_dbhz: np.ndarray,
    initial_positions: np.ndarray,
    initial_clock_biases: np.ndarray | None = None,
    grid_radius_m: float = 180.0,
    grid_step_m: float = 12.0,
    clock_bias_radius_m: float | None = None,
    clock_bias_step_m: float = 20.0,
    selected_epoch: int | None = None,
) -> dict[str, np.ndarray | CorrelogramSolution]:
    """Run the simplified correlogram solver for each epoch."""

    pseudoranges = np.asarray(pseudoranges, dtype=float)
    cn0_dbhz = np.asarray(cn0_dbhz, dtype=float)
    initial_positions = np.asarray(initial_positions, dtype=float)
    num_steps = pseudoranges.shape[0]

    positions = np.zeros((num_steps, 2), dtype=float)
    clock_biases = np.zeros(num_steps, dtype=float)
    scores = np.zeros(num_steps, dtype=float)
    selected_solution: CorrelogramSolution | None = None
    if selected_epoch is None:
        selected_epoch = int(num_steps // 2)
    selected_epoch = int(np.clip(selected_epoch, 0, max(num_steps - 1, 0)))

    if initial_clock_biases is not None:
        initial_clock_biases = np.asarray(initial_clock_biases, dtype=float)
        if initial_clock_biases.shape != (num_steps,):
            raise ValueError("initial_clock_biases must have shape (num_steps,)")

    for step in range(num_steps):
        clock_bias_search = None
        if clock_bias_radius_m is not None:
            if initial_clock_biases is None:
                raise ValueError("initial_clock_biases are required when clock_bias_radius_m is set")
            cb_step = max(float(clock_bias_step_m), 1e-6)
            cb_radius = max(float(clock_bias_radius_m), 0.0)
            center = float(initial_clock_biases[step])
            clock_bias_search = np.arange(center - cb_radius, center + cb_radius + 0.5 * cb_step, cb_step)

        solution = solve_position_pseudorange_correlogram(
            pseudoranges[step],
            satellite_positions,
            cn0_dbhz[step],
            initial_position=initial_positions[step],
            grid_radius_m=grid_radius_m,
            grid_step_m=grid_step_m,
            clock_bias_search_m=clock_bias_search,
        )
        positions[step] = solution.position
        clock_biases[step] = solution.clock_bias
        scores[step] = solution.score
        if step == selected_epoch:
            selected_solution = solution

    return {
        "positions": positions,
        "clock_biases": clock_biases,
        "scores": scores,
        "selected_epoch": np.array(selected_epoch),
        "selected_solution": selected_solution if selected_solution is not None else solution,
    }


def summarize_position_errors(true_positions: np.ndarray, estimated_positions: np.ndarray) -> dict[str, float]:
    """Return compact horizontal error metrics in meters."""

    errors = np.linalg.norm(np.asarray(estimated_positions) - np.asarray(true_positions), axis=1)
    return {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "rmse_error_m": float(np.sqrt(np.mean(errors**2))),
        "max_error_m": float(np.max(errors)),
        "final_error_m": float(errors[-1]),
    }
