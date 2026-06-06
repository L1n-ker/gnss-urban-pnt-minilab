import unittest

import numpy as np

from src.dop import compute_hdop, compute_hdop_series
from src.gnss_solver import (
    exclude_large_residuals,
    solve_position_cauchy,
    solve_position_huber,
    solve_position_least_squares,
    solve_position_weighted_least_squares,
    solve_time_series,
)
from src.ins_filter import (
    dead_reckon_from_velocity,
    gnss_ins_constant_velocity_filter,
    simulate_imu_velocity_from_truth,
)
from src.leo_aiding import generate_leo_positions
from src.sliding_window import solve_sliding_window_time_series
from src.simulation import (
    compute_true_ranges,
    generate_receiver_clock_bias,
    generate_pseudoranges,
    generate_receiver_trajectory,
    generate_satellite_positions,
    run_scenario,
)
from src.urban_environment import (
    UrbanBlock,
    classify_los_nlos,
    urban_error_map_results,
)
from src.collaborative_positioning import collaborative_positioning_results, solve_collaborative_epoch
from experiments.run_experiments import collect_summary_metrics


class DopAnalysisTests(unittest.TestCase):
    def test_hdop_is_finite_for_valid_satellite_geometry(self):
        receiver = np.array([0.0, 0.0])
        satellites = generate_satellite_positions(num_satellites=6)

        hdop = compute_hdop(receiver, satellites)

        self.assertTrue(np.isfinite(hdop))
        self.assertGreater(hdop, 0.0)

    def test_hdop_worsens_for_clustered_poor_geometry(self):
        receiver = np.array([0.0, 0.0])
        good_geometry = generate_satellite_positions(num_satellites=6)
        poor_geometry = np.array(
            [
                [20_200_000.0, 0.0],
                [20_200_000.0, 20_000.0],
                [20_200_000.0, 40_000.0],
                [20_200_000.0, 60_000.0],
                [20_200_000.0, 80_000.0],
                [20_200_000.0, 100_000.0],
            ]
        )

        self.assertGreater(compute_hdop(receiver, poor_geometry), compute_hdop(receiver, good_geometry))

    def test_leo_like_aiding_can_improve_hdop_series(self):
        trajectory = generate_receiver_trajectory(num_steps=30, seed=4)
        gnss_satellites = generate_satellite_positions(num_satellites=5)
        leo_positions = generate_leo_positions(num_steps=30, num_leo=3, seed=15)
        repeated_gnss = np.repeat(gnss_satellites[None, :, :], len(trajectory), axis=0)
        aided_sources = np.concatenate((repeated_gnss, leo_positions), axis=1)

        gnss_hdop = compute_hdop_series(trajectory, gnss_satellites)
        aided_hdop = compute_hdop_series(trajectory, aided_sources)

        self.assertLess(float(np.mean(aided_hdop)), float(np.mean(gnss_hdop)))


class RobustSolverTests(unittest.TestCase):
    def test_wls_matches_ols_when_all_sigmas_are_equal(self):
        receiver = np.array([1200.0, -800.0])
        clock_bias = 35.0
        satellites = generate_satellite_positions(num_satellites=6)
        pseudoranges = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        sigmas = np.ones(len(pseudoranges)) * 5.0

        ols = solve_position_least_squares(pseudoranges, satellites)
        wls = solve_position_weighted_least_squares(pseudoranges, satellites, sigmas)

        np.testing.assert_allclose(wls.position, ols.position, atol=1e-3)
        self.assertAlmostEqual(wls.clock_bias, ols.clock_bias, places=3)

    def test_wls_reduces_large_uncertainty_measurement_influence(self):
        receiver = np.array([900.0, 450.0])
        clock_bias = 28.0
        satellites = generate_satellite_positions(num_satellites=6)
        clean = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        biased = clean.copy()
        biased[0] += 250.0
        sigmas = np.ones(len(biased)) * 4.0
        sigmas[0] = 200.0

        ols = solve_position_least_squares(biased, satellites)
        wls = solve_position_weighted_least_squares(biased, satellites, sigmas)

        ols_error = np.linalg.norm(ols.position - receiver)
        wls_error = np.linalg.norm(wls.position - receiver)
        self.assertLess(wls_error, ols_error)

    def test_huber_solver_is_less_affected_by_outlier_than_ols(self):
        receiver = np.array([-500.0, 600.0])
        clock_bias = 42.0
        satellites = generate_satellite_positions(num_satellites=7)
        clean = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        biased = clean.copy()
        biased[[0, 1]] += np.array([220.0, 160.0])

        ols = solve_position_least_squares(biased, satellites)
        huber = solve_position_huber(biased, satellites, huber_delta=25.0)

        self.assertLess(np.linalg.norm(huber.position - receiver), np.linalg.norm(ols.position - receiver))

    def test_cauchy_solver_is_less_affected_by_outlier_than_ols(self):
        receiver = np.array([-420.0, 520.0])
        clock_bias = 37.0
        satellites = generate_satellite_positions(num_satellites=8, angular_offset=0.15)
        clean = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        biased = clean.copy()
        biased[[0, 1]] += np.array([420.0, 360.0])

        ols = solve_position_least_squares(biased, satellites)
        cauchy = solve_position_cauchy(biased, satellites, cauchy_scale=20.0)

        self.assertLess(np.linalg.norm(cauchy.position - receiver), np.linalg.norm(ols.position - receiver))

    def test_residual_exclusion_removes_clear_outliers(self):
        receiver = np.array([300.0, -300.0])
        clock_bias = 20.0
        satellites = generate_satellite_positions(num_satellites=7)
        clean = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        biased = clean.copy()
        biased[2] += 300.0

        solution, kept_mask = exclude_large_residuals(biased, satellites, residual_threshold=80.0)

        self.assertFalse(bool(kept_mask[2]))
        self.assertLess(np.linalg.norm(solution.position - receiver), 20.0)


class InsAidingTests(unittest.TestCase):
    def test_synthetic_velocity_generation_is_reproducible(self):
        trajectory = generate_receiver_trajectory(num_steps=12, seed=2)

        first = simulate_imu_velocity_from_truth(trajectory, noise_std=0.1, bias_drift_std=0.01, seed=7)
        second = simulate_imu_velocity_from_truth(trajectory, noise_std=0.1, bias_drift_std=0.01, seed=7)

        self.assertEqual(first.shape, (11, 2))
        np.testing.assert_allclose(first, second)

    def test_dead_reckoning_integrates_velocity_measurements(self):
        start = np.array([10.0, -5.0])
        velocities = np.tile(np.array([2.0, 1.0]), (5, 1))

        positions = dead_reckon_from_velocity(start, velocities, step_seconds=1.0)

        self.assertEqual(positions.shape, (6, 2))
        np.testing.assert_allclose(positions[-1], np.array([20.0, 0.0]))

    def test_gnss_ins_filter_outputs_finite_positions(self):
        trajectory = generate_receiver_trajectory(num_steps=30, seed=5)
        gnss_positions = trajectory + np.random.default_rng(5).normal(0.0, 8.0, size=trajectory.shape)
        velocities = simulate_imu_velocity_from_truth(trajectory, noise_std=0.2, bias_drift_std=0.01, seed=6)

        filtered = gnss_ins_constant_velocity_filter(
            gnss_positions,
            velocities,
            process_noise=0.8,
            measurement_noise=64.0,
        )

        self.assertEqual(filtered.shape, trajectory.shape)
        self.assertTrue(np.all(np.isfinite(filtered)))


class ScenarioUpgradeTests(unittest.TestCase):
    def test_run_scenario_exposes_hdop_metrics(self):
        result = run_scenario("GNSS + LEO-like aiding", enable_leo=True, seed=9)

        self.assertIn("hdop", result)
        self.assertIn("mean_hdop", result["metrics"])
        self.assertTrue(np.all(np.isfinite(result["hdop"])))

    def test_realistic_inputs_support_hdop_for_time_varying_sources(self):
        trajectory = generate_receiver_trajectory(num_steps=10, seed=8)
        clock = generate_receiver_clock_bias(num_steps=10, seed=9)
        satellites = generate_satellite_positions(num_satellites=6)
        leo = generate_leo_positions(num_steps=10, num_leo=2, seed=10)
        ranges = compute_true_ranges(trajectory, leo)

        self.assertEqual(ranges.shape, (10, 2))
        self.assertEqual(clock.shape, (10,))

    def test_experiment_summary_contains_expected_families(self):
        summary = collect_summary_metrics()

        self.assertIn("experiment_family", summary.columns)
        self.assertIn("mean_positioning_error_m", summary.columns)
        self.assertIn("p95_positioning_error_m", summary.columns)
        self.assertIn("mean_hdop", summary.columns)
        self.assertTrue(
            {"scenario_comparison", "robust_solver_comparison", "gnss_ins_comparison"}.issubset(
                set(summary["experiment_family"])
            )
        )

    def test_experiment_summary_includes_urban_error_map_family(self):
        summary = collect_summary_metrics()

        urban_rows = summary[summary["experiment_family"] == "urban_error_map"]

        self.assertFalse(urban_rows.empty)
        self.assertIn("OLS urban grid", set(urban_rows["case"]))
        self.assertIn("WLS urban grid", set(urban_rows["case"]))

    def test_experiment_summary_includes_sliding_and_collaborative_families(self):
        summary = collect_summary_metrics()
        families = set(summary["experiment_family"])

        self.assertIn("sliding_window_optimization", families)
        self.assertIn("collaborative_positioning", families)


class UrbanEnvironmentTests(unittest.TestCase):
    def test_geometry_blockage_classifies_nlos_measurements(self):
        receivers = np.array([[0.0, 0.0], [0.0, 200.0]])
        sources = np.array([[200.0, 0.0], [0.0, 400.0]])
        blocks = [UrbanBlock(80.0, 120.0, -40.0, 40.0)]

        nlos = classify_los_nlos(receivers, sources, blocks)

        self.assertEqual(nlos.shape, (2, 2))
        self.assertTrue(bool(nlos[0, 0]))
        self.assertFalse(bool(nlos[0, 1]))
        self.assertFalse(bool(nlos[1, 1]))

    def test_urban_error_map_outputs_error_grids_and_wls_improvement(self):
        result = urban_error_map_results(grid_size=9, seed=22)

        self.assertEqual(result["ols_error_grid"].shape, (9, 9))
        self.assertEqual(result["wls_error_grid"].shape, (9, 9))
        self.assertEqual(result["nlos_count_grid"].shape, (9, 9))
        self.assertTrue(np.all(np.isfinite(result["ols_error_grid"])))
        self.assertTrue(np.all(np.isfinite(result["wls_error_grid"])))
        self.assertGreater(float(np.max(result["nlos_count_grid"])), 0.0)
        self.assertLess(result["metrics"]["wls_mean_error_m"], result["metrics"]["ols_mean_error_m"])


class SlidingWindowOptimizationTests(unittest.TestCase):
    def test_sliding_window_outputs_finite_positions(self):
        trajectory = generate_receiver_trajectory(num_steps=24, seed=31)
        satellites = generate_satellite_positions(num_satellites=7, angular_offset=0.25)
        ranges = compute_true_ranges(trajectory, satellites)
        clock = generate_receiver_clock_bias(num_steps=24, seed=32)
        rng = np.random.default_rng(33)
        pseudoranges = generate_pseudoranges(ranges, clock, noise_std=4.0, rng=rng)
        velocities = np.diff(trajectory, axis=0) + rng.normal(0.0, 0.25, size=(23, 2))

        positions, clocks = solve_sliding_window_time_series(
            pseudoranges,
            satellites,
            odometry_deltas=velocities,
            window_size=5,
            initial_position=trajectory[0],
        )

        self.assertEqual(positions.shape, trajectory.shape)
        self.assertEqual(clocks.shape, (24,))
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(clocks)))

    def test_sliding_window_improves_degraded_interval_stability(self):
        trajectory = generate_receiver_trajectory(num_steps=36, seed=34)
        satellites = generate_satellite_positions(num_satellites=7, angular_offset=0.25)
        ranges = compute_true_ranges(trajectory, satellites)
        clock = generate_receiver_clock_bias(num_steps=36, seed=35)
        rng = np.random.default_rng(36)
        pseudoranges = generate_pseudoranges(ranges, clock, noise_std=3.5, rng=rng)
        sigmas = np.ones_like(pseudoranges) * 4.0
        degraded = slice(13, 21)
        pseudoranges[degraded, [0, 1]] += np.array([260.0, 220.0])
        sigmas[degraded, [0, 1]] = 130.0
        velocities = np.diff(trajectory, axis=0) + rng.normal(0.0, 0.18, size=(35, 2))

        single_epoch, _, _, _ = solve_time_series(
            pseudoranges,
            satellites,
            initial_position=trajectory[0],
            method="ols",
        )
        sliding, _ = solve_sliding_window_time_series(
            pseudoranges,
            satellites,
            odometry_deltas=velocities,
            measurement_sigmas=sigmas,
            window_size=6,
            initial_position=trajectory[0],
            odometry_sigma_m=0.8,
        )

        single_errors = np.linalg.norm(single_epoch[degraded] - trajectory[degraded], axis=1)
        sliding_errors = np.linalg.norm(sliding[degraded] - trajectory[degraded], axis=1)
        self.assertLess(float(np.mean(sliding_errors)), float(np.mean(single_errors)))


class CollaborativePositioningTests(unittest.TestCase):
    def test_collaborative_epoch_solution_has_finite_joint_state(self):
        agent_a = np.array([120.0, -40.0])
        agent_b = np.array([180.0, -20.0])
        satellites = generate_satellite_positions(num_satellites=7, angular_offset=0.4)
        clock_a = 35.0
        clock_b = 42.0
        pr_a = np.linalg.norm(satellites - agent_a, axis=1) + clock_a
        pr_b = np.linalg.norm(satellites - agent_b, axis=1) + clock_b
        relative_distance = float(np.linalg.norm(agent_a - agent_b))

        solution = solve_collaborative_epoch(
            pr_a,
            pr_b,
            satellites,
            relative_distance,
            initial_a=np.array([100.0, -30.0, 30.0]),
            initial_b=np.array([200.0, -10.0, 40.0]),
        )

        self.assertEqual(solution.positions.shape, (2, 2))
        self.assertEqual(solution.clock_biases.shape, (2,))
        self.assertTrue(np.all(np.isfinite(solution.positions)))

    def test_collaborative_constraint_improves_controlled_degraded_case(self):
        result = collaborative_positioning_results(num_steps=34, seed=41)

        self.assertEqual(result["collaborative_positions_a"].shape, result["true_positions_a"].shape)
        self.assertEqual(result["collaborative_positions_b"].shape, result["true_positions_b"].shape)
        self.assertLess(
            result["metrics"]["collaborative_mean_error_b_m"],
            result["metrics"]["gnss_only_mean_error_b_m"],
        )


if __name__ == "__main__":
    unittest.main()
