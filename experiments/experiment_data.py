"""Shared deterministic experiment data builders without plotting dependencies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dop import compute_hdop_series
from src.gnss_solver import solve_time_series
from src.ins_filter import (
    dead_reckon_from_velocity,
    gnss_ins_constant_velocity_filter,
    simulate_imu_velocity_from_truth,
)
from src.sliding_window import solve_sliding_window_time_series
from src.simulation import (
    compute_true_ranges,
    generate_pseudoranges,
    generate_receiver_clock_bias,
    generate_receiver_trajectory,
    generate_satellite_positions,
)
from src.leo_aiding import generate_leo_positions
from src.collaborative_positioning import collaborative_positioning_results


def robust_solver_results(seed: int = 21) -> dict[str, object]:
    """Create a fixed NLOS-biased pseudorange experiment for solver comparison."""

    rng = np.random.default_rng(seed)
    true_positions = generate_receiver_trajectory(num_steps=80, seed=seed)
    satellites = generate_satellite_positions(num_satellites=7, angular_offset=0.2)
    clock_bias = generate_receiver_clock_bias(len(true_positions), seed=seed + 100)
    clean_ranges = compute_true_ranges(true_positions, satellites)
    pseudoranges = generate_pseudoranges(clean_ranges, clock_bias, noise_std=5.0, rng=rng)

    nlos_mask = np.zeros_like(pseudoranges, dtype=bool)
    affected_satellites = [0, 1, 2]
    for sat in affected_satellites:
        events = rng.random(len(true_positions)) < 0.34
        nlos_mask[:, sat] = events
        pseudoranges[events, sat] += rng.uniform(90.0, 170.0, size=int(np.sum(events)))

    sigmas = np.ones_like(pseudoranges) * 5.0
    sigmas[nlos_mask] = 90.0

    solutions: dict[str, np.ndarray] = {}
    for method in ["ols", "wls", "huber", "cauchy", "residual_exclusion"]:
        method_sigmas = sigmas if method == "wls" else None
        positions, _, _, _ = solve_time_series(
            pseudoranges,
            satellites,
            initial_position=true_positions[0],
            method=method,
            measurement_sigmas=method_sigmas,
            huber_delta=25.0,
            cauchy_scale=25.0,
            residual_threshold=40.0,
        )
        solutions[method] = positions

    return {
        "true_positions": true_positions,
        "satellites": satellites,
        "pseudoranges": pseudoranges,
        "nlos_mask": nlos_mask,
        "sigmas": sigmas,
        "solutions": solutions,
        "hdop": compute_hdop_series(true_positions, satellites),
    }


def gnss_ins_results(seed: int = 31) -> dict[str, np.ndarray]:
    """Create a fixed GNSS degradation interval for INS/KF comparison."""

    rng = np.random.default_rng(seed)
    true_positions = generate_receiver_trajectory(num_steps=80, seed=seed)
    gnss_positions = true_positions + rng.normal(0.0, 7.0, size=true_positions.shape)
    gnss_positions[32:48] += np.column_stack((np.linspace(35.0, 140.0, 16), np.linspace(-20.0, -90.0, 16)))
    velocities = simulate_imu_velocity_from_truth(
        true_positions,
        noise_std=0.35,
        bias=np.array([0.05, -0.03]),
        bias_drift_std=0.015,
        seed=seed + 1,
    )
    ins_only = dead_reckon_from_velocity(true_positions[0], velocities)
    kf_positions = gnss_ins_constant_velocity_filter(
        gnss_positions,
        velocities,
        process_noise=0.8,
        measurement_noise=49.0,
        velocity_noise=2.0,
    )
    return {
        "true_positions": true_positions,
        "gnss_positions": gnss_positions,
        "ins_only": ins_only,
        "kf_positions": kf_positions,
    }


def clean_leo_comparison_results(seed: int = 44, num_steps: int = 80, num_leo: int = 3) -> dict[str, object]:
    """Compare clean GNSS-only and clean GNSS + LEO-like aiding.

    Both cases use the same receiver trajectory, clock bias, random seed, and
    clean Gaussian measurement-noise level. The only intended difference is the
    additional LEO-like ranging geometry.
    """

    rng = np.random.default_rng(seed)
    true_positions = generate_receiver_trajectory(num_steps=num_steps, seed=seed)
    gnss_satellites = generate_satellite_positions(num_satellites=6)
    leo_positions = generate_leo_positions(num_steps=num_steps, num_leo=num_leo, seed=seed + 200)
    clock_bias = generate_receiver_clock_bias(num_steps, seed=seed + 100)

    gnss_ranges = compute_true_ranges(true_positions, gnss_satellites)
    gnss_pseudoranges = generate_pseudoranges(gnss_ranges, clock_bias, noise_std=5.0, rng=rng)
    gnss_positions, _, _, _ = solve_time_series(
        gnss_pseudoranges,
        gnss_satellites,
        initial_position=true_positions[0],
    )

    leo_ranges = compute_true_ranges(true_positions, leo_positions)
    leo_pseudoranges = generate_pseudoranges(leo_ranges, clock_bias, noise_std=3.0, rng=rng)
    repeated_gnss = np.repeat(gnss_satellites[None, :, :], num_steps, axis=0)
    aided_sources = np.concatenate((repeated_gnss, leo_positions), axis=1)
    aided_pseudoranges = np.concatenate((gnss_pseudoranges, leo_pseudoranges), axis=1)
    aided_positions, _, _, _ = solve_time_series(
        aided_pseudoranges,
        aided_sources,
        initial_position=true_positions[0],
    )

    gnss_errors = np.linalg.norm(gnss_positions - true_positions, axis=1)
    aided_errors = np.linalg.norm(aided_positions - true_positions, axis=1)

    return {
        "true_positions": true_positions,
        "gnss_satellites": gnss_satellites,
        "leo_positions": leo_positions,
        "gnss_only": {
            "positions": gnss_positions,
            "errors": gnss_errors,
            "hdop": compute_hdop_series(true_positions, gnss_satellites),
        },
        "gnss_leo": {
            "positions": aided_positions,
            "errors": aided_errors,
            "hdop": compute_hdop_series(true_positions, aided_sources),
        },
    }


def sliding_window_results(seed: int = 52, num_steps: int = 80) -> dict[str, object]:
    """Create a controlled degraded interval for sliding-window comparison."""

    rng = np.random.default_rng(seed)
    true_positions = generate_receiver_trajectory(num_steps=num_steps, seed=seed)
    satellites = generate_satellite_positions(num_satellites=7, angular_offset=0.25)
    clock_bias = generate_receiver_clock_bias(num_steps, seed=seed + 100)
    ranges = compute_true_ranges(true_positions, satellites)
    pseudoranges = generate_pseudoranges(ranges, clock_bias, noise_std=3.5, rng=rng)
    sigmas = np.ones_like(pseudoranges) * 4.0

    degraded_start = int(0.35 * num_steps)
    degraded_stop = int(0.58 * num_steps)
    pseudoranges[degraded_start:degraded_stop, [0, 1]] += np.array([260.0, 220.0])
    sigmas[degraded_start:degraded_stop, [0, 1]] = 130.0
    odometry_deltas = np.diff(true_positions, axis=0) + rng.normal(0.0, 0.2, size=(num_steps - 1, 2))

    single_epoch, _, _, _ = solve_time_series(
        pseudoranges,
        satellites,
        initial_position=true_positions[0],
        method="ols",
    )
    sliding_positions, sliding_clocks = solve_sliding_window_time_series(
        pseudoranges,
        satellites,
        odometry_deltas=odometry_deltas,
        measurement_sigmas=sigmas,
        window_size=7,
        initial_position=true_positions[0],
        odometry_sigma_m=0.9,
    )
    single_errors = np.linalg.norm(single_epoch - true_positions, axis=1)
    sliding_errors = np.linalg.norm(sliding_positions - true_positions, axis=1)
    degraded_slice = slice(degraded_start, degraded_stop)

    return {
        "true_positions": true_positions,
        "satellites": satellites,
        "pseudoranges": pseudoranges,
        "sigmas": sigmas,
        "odometry_deltas": odometry_deltas,
        "single_epoch_positions": single_epoch,
        "sliding_positions": sliding_positions,
        "sliding_clocks": sliding_clocks,
        "single_epoch_errors": single_errors,
        "sliding_errors": sliding_errors,
        "degraded_start": degraded_start,
        "degraded_stop": degraded_stop,
        "metrics": {
            "single_epoch_mean_error_m": float(np.mean(single_errors)),
            "sliding_window_mean_error_m": float(np.mean(sliding_errors)),
            "single_epoch_degraded_mean_error_m": float(np.mean(single_errors[degraded_slice])),
            "sliding_window_degraded_mean_error_m": float(np.mean(sliding_errors[degraded_slice])),
            "sliding_window_rmse_error_m": float(np.sqrt(np.mean(sliding_errors**2))),
        },
    }


def collaborative_results(seed: int = 61, num_steps: int = 80) -> dict[str, object]:
    """Create a deterministic two-agent collaborative-positioning experiment."""

    return collaborative_positioning_results(num_steps=num_steps, seed=seed)
