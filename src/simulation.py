"""Scenario orchestration for GNSS Robustness MiniLab."""

from __future__ import annotations

from typing import Any

import numpy as np

from .anomaly_detection import detect_residual_anomalies, robust_residual_threshold
from .dop import compute_hdop_series
from .error_models import apply_multipath_nlos, apply_spoofing_drift
from .gnss_solver import solve_time_series
from .ins_filter import dead_reckon, fuse_gnss_ins
from .leo_aiding import generate_leo_positions


DEFAULT_SCENARIOS = [
    "Ideal GNSS",
    "Multipath/NLOS",
    "Spoofing-like drift",
    "GNSS + INS",
    "GNSS + LEO-like aiding",
    "GNSS + INS + LEO-like aiding",
]


def generate_receiver_trajectory(num_steps: int = 80, seed: int | None = None) -> np.ndarray:
    """Generate a smooth 2D receiver trajectory in meters."""

    num_steps = int(num_steps)
    if num_steps <= 0:
        return np.zeros((0, 2), dtype=float)

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, num_steps)
    phase = rng.uniform(-0.2, 0.2)
    lateral_scale = 260.0 + rng.uniform(-25.0, 25.0)

    x = -900.0 + 2_000.0 * t + 80.0 * np.sin(2.0 * np.pi * (t + phase))
    y = -250.0 + 700.0 * t + lateral_scale * np.sin(np.pi * (t + 0.15))
    return np.column_stack((x, y))


def generate_satellite_positions(
    num_satellites: int = 6,
    radius: float = 20_200_000.0,
    angular_offset: float = 0.35,
) -> np.ndarray:
    """Generate fixed 2D GNSS satellite positions around the receiver area."""

    num_satellites = int(num_satellites)
    if num_satellites <= 0:
        return np.zeros((0, 2), dtype=float)

    angles = angular_offset + np.linspace(0.0, 2.0 * np.pi, num_satellites, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def compute_true_ranges(receiver_positions: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Compute geometric ranges from each receiver epoch to each source."""

    receiver_positions = np.asarray(receiver_positions, dtype=float)
    source_positions = np.asarray(source_positions, dtype=float)

    if source_positions.ndim == 2:
        deltas = receiver_positions[:, None, :] - source_positions[None, :, :]
    elif source_positions.ndim == 3:
        deltas = receiver_positions[:, None, :] - source_positions
    else:
        raise ValueError("source_positions must be fixed 2D or time-varying 3D")

    return np.linalg.norm(deltas, axis=2)


def generate_receiver_clock_bias(num_steps: int, seed: int | None = None) -> np.ndarray:
    """Create a small time-varying receiver clock bias in meters."""

    rng = np.random.default_rng(seed)
    steps = np.arange(num_steps, dtype=float)
    smooth_bias = 45.0 + 7.0 * np.sin(2.0 * np.pi * steps / max(num_steps, 1))
    random_walk = np.cumsum(rng.normal(0.0, 0.08, size=num_steps))
    return smooth_bias + random_walk


def generate_pseudoranges(
    true_ranges: np.ndarray,
    clock_bias_m: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add receiver clock bias and Gaussian measurement noise."""

    true_ranges = np.asarray(true_ranges, dtype=float)
    clock_bias_m = np.asarray(clock_bias_m, dtype=float)
    noise = rng.normal(0.0, max(float(noise_std), 0.0), size=true_ranges.shape)
    return true_ranges + clock_bias_m[:, None] + noise


def _scenario_uses_multipath(scenario: str) -> bool:
    scenario_lower = scenario.lower()
    return (
        "multipath" in scenario_lower
        or "nlos" in scenario_lower
        or "ins" in scenario_lower
        or "leo" in scenario_lower
    ) and "ideal" not in scenario_lower


def _scenario_uses_spoofing(scenario: str) -> bool:
    return "spoof" in scenario.lower()


def _time_varying_sources(fixed_sources: np.ndarray, moving_sources: np.ndarray | None) -> np.ndarray:
    if moving_sources is None or moving_sources.shape[1] == 0:
        return fixed_sources

    repeated_fixed = np.repeat(fixed_sources[None, :, :], moving_sources.shape[0], axis=0)
    return np.concatenate((repeated_fixed, moving_sources), axis=1)


def _compute_metrics(
    true_positions: np.ndarray,
    estimated_positions: np.ndarray,
    residual_rms: np.ndarray,
    anomaly_flags: np.ndarray,
    converged: np.ndarray,
    hdop: np.ndarray | None = None,
) -> dict[str, float]:
    errors = np.linalg.norm(estimated_positions - true_positions, axis=1)
    metrics = {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "rmse_error_m": float(np.sqrt(np.mean(errors**2))),
        "p95_error_m": float(np.percentile(errors, 95.0)),
        "max_error_m": float(np.max(errors)),
        "final_error_m": float(errors[-1]),
        "mean_residual_rms_m": float(np.mean(residual_rms)),
        "max_residual_rms_m": float(np.max(residual_rms)),
        "anomaly_count": float(np.sum(anomaly_flags)),
        "convergence_rate_pct": float(100.0 * np.mean(converged)),
    }
    if hdop is not None:
        metrics["mean_hdop"] = float(np.mean(hdop))
        metrics["max_hdop"] = float(np.max(hdop))
    return metrics


def run_scenario(
    scenario: str = "Ideal GNSS",
    num_steps: int = 80,
    measurement_noise_std: float = 5.0,
    multipath_bias_level: float = 40.0,
    nlos_probability: float = 0.2,
    spoofing_start_time: int = 45,
    spoofing_drift_strength: float = 2.0,
    enable_ins: bool = False,
    enable_leo: bool = False,
    num_leo: int = 2,
    seed: int = 4,
) -> dict[str, Any]:
    """Run one complete educational positioning scenario."""

    rng = np.random.default_rng(seed)
    scenario_lower = scenario.lower()
    use_ins = bool(enable_ins or "ins" in scenario_lower)
    use_leo = bool(enable_leo or "leo" in scenario_lower)

    true_positions = generate_receiver_trajectory(num_steps=num_steps, seed=seed)
    gnss_satellites = generate_satellite_positions(num_satellites=6)
    clock_bias = generate_receiver_clock_bias(num_steps, seed=seed + 100)

    gnss_ranges = compute_true_ranges(true_positions, gnss_satellites)
    gnss_pseudoranges = generate_pseudoranges(
        gnss_ranges,
        clock_bias,
        noise_std=measurement_noise_std,
        rng=rng,
    )

    if _scenario_uses_multipath(scenario):
        gnss_pseudoranges = apply_multipath_nlos(
            gnss_pseudoranges,
            bias_level=multipath_bias_level,
            nlos_probability=nlos_probability,
            rng=rng,
            affected_satellites=[0, 1, 2],
        )

    if _scenario_uses_spoofing(scenario):
        gnss_pseudoranges = apply_spoofing_drift(
            gnss_pseudoranges,
            start_step=spoofing_start_time,
            drift_strength=spoofing_drift_strength,
            affected_satellites=[0, 1],
        )

    leo_positions = None
    all_sources = gnss_satellites
    all_pseudoranges = gnss_pseudoranges

    if use_leo:
        leo_positions = generate_leo_positions(
            num_steps=num_steps,
            num_leo=num_leo,
            seed=seed + 200,
        )
        leo_ranges = compute_true_ranges(true_positions, leo_positions)
        leo_pseudoranges = generate_pseudoranges(
            leo_ranges,
            clock_bias,
            noise_std=max(1.0, measurement_noise_std * 0.6),
            rng=rng,
        )
        all_sources = _time_varying_sources(gnss_satellites, leo_positions)
        all_pseudoranges = np.concatenate((gnss_pseudoranges, leo_pseudoranges), axis=1)

    raw_positions, estimated_clock, residual_rms, converged = solve_time_series(
        all_pseudoranges,
        all_sources,
        initial_position=true_positions[0],
    )
    hdop = compute_hdop_series(true_positions, all_sources)

    ins_positions = None
    estimated_positions = raw_positions.copy()
    if use_ins:
        ins_positions = dead_reckon(
            true_positions,
            step_seconds=1.0,
            noise_std=0.55,
            seed=seed + 300,
        )
        gnss_weight = 0.55 if not use_leo else 0.65
        estimated_positions = fuse_gnss_ins(raw_positions, ins_positions, gnss_weight=gnss_weight)

    residual_threshold = max(robust_residual_threshold(residual_rms), measurement_noise_std * 2.5)
    anomaly_flags = detect_residual_anomalies(residual_rms, threshold=residual_threshold)
    errors = np.linalg.norm(estimated_positions - true_positions, axis=1)
    metrics = _compute_metrics(
        true_positions,
        estimated_positions,
        residual_rms,
        anomaly_flags,
        converged,
        hdop,
    )

    return {
        "scenario": scenario,
        "true_positions": true_positions,
        "estimated_positions": estimated_positions,
        "raw_gnss_positions": raw_positions,
        "ins_positions": ins_positions,
        "gnss_satellites": gnss_satellites,
        "leo_positions": leo_positions,
        "pseudoranges": all_pseudoranges,
        "clock_bias_true": clock_bias,
        "clock_bias_estimated": estimated_clock,
        "position_errors": errors,
        "residual_rms": residual_rms,
        "residual_threshold": residual_threshold,
        "anomaly_flags": anomaly_flags,
        "converged": converged,
        "hdop": hdop,
        "metrics": metrics,
    }


def run_comparison(
    measurement_noise_std: float = 5.0,
    multipath_bias_level: float = 40.0,
    nlos_probability: float = 0.2,
    spoofing_start_time: int = 45,
    spoofing_drift_strength: float = 2.0,
    num_leo: int = 2,
    seed: int = 4,
    num_steps: int = 80,
) -> list[dict[str, float | str]]:
    """Run the standard scenario set and return compact metric rows."""

    rows: list[dict[str, float | str]] = []
    for scenario in DEFAULT_SCENARIOS:
        result = run_scenario(
            scenario=scenario,
            num_steps=num_steps,
            measurement_noise_std=measurement_noise_std,
            multipath_bias_level=multipath_bias_level,
            nlos_probability=nlos_probability,
            spoofing_start_time=spoofing_start_time,
            spoofing_drift_strength=spoofing_drift_strength,
            enable_ins="INS" in scenario,
            enable_leo="LEO" in scenario,
            num_leo=num_leo,
            seed=seed,
        )
        row: dict[str, float | str] = {"scenario": scenario}
        row.update(result["metrics"])
        rows.append(row)

    return rows
