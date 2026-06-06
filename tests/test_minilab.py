import unittest

import numpy as np

from src.anomaly_detection import detect_residual_anomalies
from src.error_models import apply_multipath_nlos, apply_spoofing_drift
from src.gnss_solver import solve_position_least_squares
from src.ins_filter import dead_reckon, fuse_gnss_ins
from src.leo_aiding import generate_leo_positions
from src.simulation import (
    generate_receiver_trajectory,
    generate_satellite_positions,
    run_scenario,
)


class MiniLabCoreTests(unittest.TestCase):
    def test_trajectory_is_deterministic_and_two_dimensional(self):
        first = generate_receiver_trajectory(num_steps=8, seed=7)
        second = generate_receiver_trajectory(num_steps=8, seed=7)

        self.assertEqual(first.shape, (8, 2))
        np.testing.assert_allclose(first, second)
        self.assertGreater(np.linalg.norm(first[-1] - first[0]), 100.0)

    def test_satellite_geometry_surrounds_receiver_area(self):
        satellites = generate_satellite_positions(num_satellites=6, radius=20_200_000.0)

        self.assertEqual(satellites.shape, (6, 2))
        self.assertTrue(np.all(np.linalg.norm(satellites, axis=1) > 10_000_000.0))

    def test_least_squares_recovers_position_and_clock_bias(self):
        receiver = np.array([1200.0, -800.0])
        clock_bias = 35.0
        satellites = generate_satellite_positions(num_satellites=6)
        pseudoranges = np.linalg.norm(satellites - receiver, axis=1) + clock_bias

        solution = solve_position_least_squares(pseudoranges, satellites)

        np.testing.assert_allclose(solution.position, receiver, atol=1e-3)
        self.assertAlmostEqual(solution.clock_bias, clock_bias, places=3)
        self.assertLess(solution.residual_rms, 1e-3)
        self.assertTrue(solution.converged)

    def test_error_models_add_positive_bias_and_drift(self):
        base = np.full((6, 5), 100.0)
        multipath = apply_multipath_nlos(
            base,
            bias_level=25.0,
            nlos_probability=1.0,
            rng=np.random.default_rng(3),
        )
        drift = apply_spoofing_drift(
            base,
            start_step=3,
            drift_strength=4.0,
            affected_satellites=[0, 1],
        )

        self.assertTrue(np.all(multipath >= base))
        np.testing.assert_allclose(drift[:3], base[:3])
        self.assertGreater(drift[-1, 0], drift[3, 0])
        np.testing.assert_allclose(drift[:, 2:], base[:, 2:])

    def test_residual_anomaly_detection_flags_large_residuals(self):
        residuals = np.array([1.0, 1.5, 2.0, 12.0, 20.0])
        flags = detect_residual_anomalies(residuals, threshold=6.0)

        np.testing.assert_array_equal(flags, np.array([False, False, False, True, True]))

    def test_dead_reckoning_and_weighted_fusion_shapes(self):
        trajectory = generate_receiver_trajectory(num_steps=10, seed=2)
        ins = dead_reckon(trajectory, step_seconds=1.0, noise_std=0.0, seed=5)
        noisy_gnss = trajectory + 10.0
        fused = fuse_gnss_ins(noisy_gnss, ins, gnss_weight=0.75)

        self.assertEqual(ins.shape, trajectory.shape)
        self.assertEqual(fused.shape, trajectory.shape)
        np.testing.assert_allclose(ins, trajectory, atol=1e-6)
        np.testing.assert_allclose(fused, 0.75 * noisy_gnss + 0.25 * ins)

    def test_leo_positions_are_time_varying(self):
        leo = generate_leo_positions(num_steps=12, num_leo=3, seed=11)

        self.assertEqual(leo.shape, (12, 3, 2))
        self.assertGreater(np.linalg.norm(leo[0, 0] - leo[-1, 0]), 1_000.0)

    def test_leo_and_ins_aiding_improves_noisy_scenario(self):
        gnss_only = run_scenario(
            scenario="Multipath/NLOS",
            num_steps=40,
            measurement_noise_std=12.0,
            multipath_bias_level=80.0,
            nlos_probability=0.35,
            enable_ins=False,
            enable_leo=False,
            seed=42,
        )
        aided = run_scenario(
            scenario="GNSS + INS + LEO-like aiding",
            num_steps=40,
            measurement_noise_std=12.0,
            multipath_bias_level=80.0,
            nlos_probability=0.35,
            enable_ins=True,
            enable_leo=True,
            num_leo=3,
            seed=42,
        )

        self.assertLess(
            aided["metrics"]["mean_error_m"],
            gnss_only["metrics"]["mean_error_m"],
        )
        self.assertIn("estimated_positions", aided)
        self.assertIn("residual_rms", aided)


if __name__ == "__main__":
    unittest.main()
