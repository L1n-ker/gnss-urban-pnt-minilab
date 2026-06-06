"""INS-like prediction and simple GNSS/INS fusion.

The goal is educational clarity, not a high-fidelity inertial navigation model.
We infer a velocity profile from the true simulated trajectory, perturb it with
small random velocity errors, and integrate it forward. This mimics how dead
reckoning can bridge noisy GNSS epochs while still drifting over time.
"""

from __future__ import annotations

import numpy as np


def simulate_imu_velocity_from_truth(
    true_positions: np.ndarray,
    step_seconds: float = 1.0,
    noise_std: float = 0.8,
    bias: np.ndarray | None = None,
    bias_drift_std: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Create synthetic odometry/IMU-like velocity measurements from truth.

    The simulator uses the known trajectory only to synthesize beginner-friendly
    velocity measurements with noise and slowly drifting bias. Downstream dead
    reckoning integrates these measurements rather than reading true positions.
    """

    true_positions = np.asarray(true_positions, dtype=float)
    if true_positions.ndim != 2 or true_positions.shape[1] != 2:
        raise ValueError("true_positions must have shape (num_steps, 2)")
    if len(true_positions) < 2:
        return np.zeros((0, 2), dtype=float)

    rng = np.random.default_rng(seed)
    true_velocities = np.diff(true_positions, axis=0) / float(step_seconds)
    if bias is None:
        bias_vector = np.zeros(2, dtype=float)
    else:
        bias_vector = np.asarray(bias, dtype=float)
        if bias_vector.shape != (2,):
            raise ValueError("bias must have shape (2,)")

    drift = np.cumsum(rng.normal(0.0, bias_drift_std, size=true_velocities.shape), axis=0)
    white_noise = rng.normal(0.0, noise_std, size=true_velocities.shape)
    return true_velocities + bias_vector + drift + white_noise


def dead_reckon_from_velocity(
    initial_position: np.ndarray,
    velocity_measurements: np.ndarray,
    step_seconds: float = 1.0,
) -> np.ndarray:
    """Integrate synthetic velocity/odometry measurements into positions."""

    initial_position = np.asarray(initial_position, dtype=float)
    velocity_measurements = np.asarray(velocity_measurements, dtype=float)
    if initial_position.shape != (2,):
        raise ValueError("initial_position must have shape (2,)")
    if velocity_measurements.ndim != 2 or velocity_measurements.shape[1] != 2:
        raise ValueError("velocity_measurements must have shape (num_steps - 1, 2)")

    predicted = np.zeros((len(velocity_measurements) + 1, 2), dtype=float)
    predicted[0] = initial_position
    for step, velocity in enumerate(velocity_measurements, start=1):
        predicted[step] = predicted[step - 1] + velocity * float(step_seconds)

    return predicted


def dead_reckon(
    reference_positions: np.ndarray,
    step_seconds: float = 1.0,
    noise_std: float = 0.8,
    seed: int | None = None,
) -> np.ndarray:
    """Compatibility wrapper for the original INS-like dead reckoning.

    This now generates synthetic velocity measurements first, then integrates
    those measurements. With zero noise and zero drift, it reproduces the input
    trajectory, preserving the old tests and GUI behavior.
    """

    reference_positions = np.asarray(reference_positions, dtype=float)
    if len(reference_positions) == 0:
        return reference_positions.copy()
    velocities = simulate_imu_velocity_from_truth(
        reference_positions,
        step_seconds=step_seconds,
        noise_std=noise_std,
        bias_drift_std=0.0,
        seed=seed,
    )
    return dead_reckon_from_velocity(reference_positions[0], velocities, step_seconds=step_seconds)


def fuse_gnss_ins(
    gnss_positions: np.ndarray,
    ins_positions: np.ndarray,
    gnss_weight: float = 0.65,
) -> np.ndarray:
    """Blend GNSS and INS estimates with a transparent weighted average."""

    gnss_positions = np.asarray(gnss_positions, dtype=float)
    ins_positions = np.asarray(ins_positions, dtype=float)

    if gnss_positions.shape != ins_positions.shape:
        raise ValueError("gnss_positions and ins_positions must have matching shapes")

    gnss_weight = float(np.clip(gnss_weight, 0.0, 1.0))
    return gnss_weight * gnss_positions + (1.0 - gnss_weight) * ins_positions


def constant_velocity_kalman_filter(
    measurements: np.ndarray,
    process_noise: float = 1.0,
    measurement_noise: float = 25.0,
    step_seconds: float = 1.0,
) -> np.ndarray:
    """Small 2D constant-velocity Kalman filter for optional experimentation."""

    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim != 2 or measurements.shape[1] != 2:
        raise ValueError("measurements must have shape (num_steps, 2)")
    if len(measurements) == 0:
        return measurements.copy()

    dt = float(step_seconds)
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    process_covariance = np.eye(4) * float(process_noise)
    measurement_covariance = np.eye(2) * float(measurement_noise)

    state = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    covariance = np.eye(4) * 100.0
    filtered = np.zeros_like(measurements)

    for step, measurement in enumerate(measurements):
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_covariance

        innovation = measurement - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + measurement_covariance
        gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        state = state + gain @ innovation
        covariance = (np.eye(4) - gain @ observation) @ covariance
        filtered[step] = state[:2]

    return filtered


def gnss_ins_constant_velocity_filter(
    gnss_positions: np.ndarray,
    velocity_measurements: np.ndarray,
    process_noise: float = 1.0,
    measurement_noise: float = 25.0,
    velocity_noise: float = 4.0,
    step_seconds: float = 1.0,
) -> np.ndarray:
    """Toy GNSS/INS Kalman filter using position and synthetic velocity data.

    The state is ``[px, py, vx, vy]``. The prediction step uses the measured
    velocity as a control-like input for position propagation. The update step
    uses GNSS position plus the same velocity measurement as a simple odometry
    observation. This is a transparent teaching baseline, not a navigation-grade
    inertial mechanization.
    """

    gnss_positions = np.asarray(gnss_positions, dtype=float)
    velocity_measurements = np.asarray(velocity_measurements, dtype=float)

    if gnss_positions.ndim != 2 or gnss_positions.shape[1] != 2:
        raise ValueError("gnss_positions must have shape (num_steps, 2)")
    if velocity_measurements.shape != (max(len(gnss_positions) - 1, 0), 2):
        raise ValueError("velocity_measurements must have shape (num_steps - 1, 2)")
    if len(gnss_positions) == 0:
        return gnss_positions.copy()

    dt = float(step_seconds)
    state = np.array([gnss_positions[0, 0], gnss_positions[0, 1], 0.0, 0.0], dtype=float)
    covariance = np.eye(4) * 100.0
    process_covariance = np.eye(4) * float(process_noise)
    observation = np.eye(4)
    measurement_covariance = np.diag(
        [float(measurement_noise), float(measurement_noise), float(velocity_noise), float(velocity_noise)]
    )
    filtered = np.zeros_like(gnss_positions)
    filtered[0] = state[:2]

    for step in range(1, len(gnss_positions)):
        velocity = velocity_measurements[step - 1]
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        state = transition @ state
        state[:2] = filtered[step - 1] + velocity * dt
        state[2:] = velocity
        covariance = transition @ covariance @ transition.T + process_covariance

        measurement = np.array([gnss_positions[step, 0], gnss_positions[step, 1], velocity[0], velocity[1]])
        innovation = measurement - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + measurement_covariance
        gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        state = state + gain @ innovation
        covariance = (np.eye(4) - gain @ observation) @ covariance
        filtered[step] = state[:2]

    return filtered
