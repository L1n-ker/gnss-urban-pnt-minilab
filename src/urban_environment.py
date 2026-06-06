"""Simplified 2D urban LOS/NLOS and error-map utilities.

The geometry here is intentionally educational: buildings are axis-aligned
rectangles in a local 2D plane, and a measurement is labeled NLOS when the
receiver-to-source line segment intersects a building footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dop import compute_hdop
from .gnss_solver import solve_position_least_squares, solve_position_weighted_least_squares
from .simulation import generate_satellite_positions


@dataclass(frozen=True)
class UrbanBlock:
    """Axis-aligned rectangular building footprint in local East-North meters."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def corners(self) -> np.ndarray:
        return np.array(
            [
                [self.min_x, self.min_y],
                [self.max_x, self.min_y],
                [self.max_x, self.max_y],
                [self.min_x, self.max_y],
            ],
            dtype=float,
        )


def default_urban_blocks() -> list[UrbanBlock]:
    """Return a small synthetic urban canyon layout."""

    return [
        UrbanBlock(-620.0, -330.0, -250.0, 430.0),
        UrbanBlock(-120.0, 120.0, -520.0, 70.0),
        UrbanBlock(310.0, 620.0, -120.0, 520.0),
        UrbanBlock(-40.0, 260.0, 260.0, 610.0),
    ]


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
        and abs(_orientation(a, b, c)) <= 1e-9
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0):
        return True
    return _on_segment(a, c, b) or _on_segment(a, d, b) or _on_segment(c, a, d) or _on_segment(c, b, d)


def _point_in_block(point: np.ndarray, block: UrbanBlock) -> bool:
    return block.min_x <= point[0] <= block.max_x and block.min_y <= point[1] <= block.max_y


def _segment_intersects_block(start: np.ndarray, end: np.ndarray, block: UrbanBlock) -> bool:
    if _point_in_block(start, block) or _point_in_block(end, block):
        return True

    corners = block.corners()
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    return any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges)


def classify_los_nlos(
    receiver_positions: np.ndarray,
    source_positions: np.ndarray,
    blocks: list[UrbanBlock] | None = None,
) -> np.ndarray:
    """Return a boolean NLOS mask for receiver-source measurements."""

    receivers = np.asarray(receiver_positions, dtype=float)
    sources = np.asarray(source_positions, dtype=float)
    urban_blocks = default_urban_blocks() if blocks is None else blocks

    if receivers.ndim == 1:
        receivers = receivers.reshape(1, 2)
    if receivers.ndim != 2 or receivers.shape[1] != 2:
        raise ValueError("receiver_positions must have shape (num_positions, 2)")
    if sources.ndim not in (2, 3) or sources.shape[-1] != 2:
        raise ValueError("source_positions must have shape (num_sources, 2) or (num_positions, num_sources, 2)")
    if sources.ndim == 3 and sources.shape[0] != receivers.shape[0]:
        raise ValueError("time-varying source_positions must match the number of receiver positions")

    num_positions = receivers.shape[0]
    num_sources = sources.shape[0] if sources.ndim == 2 else sources.shape[1]
    nlos = np.zeros((num_positions, num_sources), dtype=bool)

    for pos_index, receiver in enumerate(receivers):
        epoch_sources = sources if sources.ndim == 2 else sources[pos_index]
        for source_index, source in enumerate(epoch_sources):
            nlos[pos_index, source_index] = any(
                _segment_intersects_block(receiver, source, block) for block in urban_blocks
            )

    return nlos


def measurement_sigmas_from_nlos(
    nlos_mask: np.ndarray,
    los_sigma_m: float = 4.0,
    nlos_sigma_m: float = 95.0,
) -> np.ndarray:
    """Map LOS/NLOS labels to measurement standard deviations."""

    nlos = np.asarray(nlos_mask, dtype=bool)
    sigmas = np.full(nlos.shape, float(los_sigma_m), dtype=float)
    sigmas[nlos] = float(nlos_sigma_m)
    return sigmas


def apply_urban_nlos_effects(
    geometric_ranges: np.ndarray,
    clock_bias_m: float,
    nlos_mask: np.ndarray,
    rng: np.random.Generator,
    los_sigma_m: float = 4.0,
    nlos_sigma_m: float = 95.0,
    nlos_bias_min_m: float = 150.0,
    nlos_bias_max_m: float = 320.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create pseudoranges with larger positive bias/variance for NLOS links."""

    ranges = np.asarray(geometric_ranges, dtype=float)
    nlos = np.asarray(nlos_mask, dtype=bool)
    sigmas = measurement_sigmas_from_nlos(nlos, los_sigma_m=los_sigma_m, nlos_sigma_m=nlos_sigma_m)
    pseudoranges = ranges + float(clock_bias_m) + rng.normal(0.0, sigmas)
    if np.any(nlos):
        pseudoranges[nlos] += rng.uniform(nlos_bias_min_m, nlos_bias_max_m, size=int(np.sum(nlos)))
    return pseudoranges, sigmas


def urban_error_map_results(
    grid_size: int = 21,
    seed: int = 22,
    blocks: list[UrbanBlock] | None = None,
) -> dict[str, np.ndarray | dict[str, float] | list[UrbanBlock]]:
    """Evaluate OLS and WLS over a synthetic urban grid."""

    grid_size = int(max(grid_size, 3))
    rng = np.random.default_rng(seed)
    urban_blocks = default_urban_blocks() if blocks is None else blocks
    x_grid = np.linspace(-850.0, 850.0, grid_size)
    y_grid = np.linspace(-650.0, 750.0, grid_size)
    satellites = generate_satellite_positions(num_satellites=8, angular_offset=0.2)

    ols_error = np.zeros((grid_size, grid_size), dtype=float)
    wls_error = np.zeros_like(ols_error)
    hdop_grid = np.zeros_like(ols_error)
    nlos_count = np.zeros_like(ols_error)

    for row_index, y_value in enumerate(y_grid):
        for col_index, x_value in enumerate(x_grid):
            receiver = np.array([x_value, y_value], dtype=float)
            ranges = np.linalg.norm(satellites - receiver, axis=1)
            nlos = classify_los_nlos(receiver, satellites, urban_blocks)[0]
            pseudoranges, sigmas = apply_urban_nlos_effects(
                ranges,
                clock_bias_m=38.0,
                nlos_mask=nlos,
                rng=rng,
            )
            initial_guess = np.array([0.0, 0.0, 0.0], dtype=float)
            ols = solve_position_least_squares(pseudoranges, satellites, initial_guess=initial_guess)
            wls = solve_position_weighted_least_squares(pseudoranges, satellites, sigmas, initial_guess=initial_guess)
            ols_error[row_index, col_index] = float(np.linalg.norm(ols.position - receiver))
            wls_error[row_index, col_index] = float(np.linalg.norm(wls.position - receiver))
            hdop_grid[row_index, col_index] = compute_hdop(receiver, satellites)
            nlos_count[row_index, col_index] = float(np.sum(nlos))

    metrics = {
        "ols_mean_error_m": float(np.mean(ols_error)),
        "wls_mean_error_m": float(np.mean(wls_error)),
        "ols_rmse_error_m": float(np.sqrt(np.mean(ols_error**2))),
        "wls_rmse_error_m": float(np.sqrt(np.mean(wls_error**2))),
        "mean_nlos_count": float(np.mean(nlos_count)),
        "mean_hdop": float(np.mean(hdop_grid)),
        "max_hdop": float(np.max(hdop_grid)),
    }
    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "satellites": satellites,
        "blocks": urban_blocks,
        "ols_error_grid": ols_error,
        "wls_error_grid": wls_error,
        "hdop_grid": hdop_grid,
        "nlos_count_grid": nlos_count,
        "metrics": metrics,
    }


def urban_los_nlos_demo_results(
    num_steps: int = 80,
    seed: int = 22,
    blocks: list[UrbanBlock] | None = None,
) -> dict[str, np.ndarray | list[UrbanBlock]]:
    """Generate a route-level LOS/NLOS labeling demonstration."""

    from .simulation import generate_receiver_trajectory

    trajectory = generate_receiver_trajectory(num_steps=num_steps, seed=seed)
    satellites = generate_satellite_positions(num_satellites=8, angular_offset=0.2)
    urban_blocks = default_urban_blocks() if blocks is None else blocks
    nlos = classify_los_nlos(trajectory, satellites, urban_blocks)
    sigmas = measurement_sigmas_from_nlos(nlos)
    return {
        "trajectory": trajectory,
        "satellites": satellites,
        "blocks": urban_blocks,
        "nlos_mask": nlos,
        "measurement_sigmas": sigmas,
        "nlos_count": np.sum(nlos, axis=1),
    }
