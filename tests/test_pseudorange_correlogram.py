import unittest

import numpy as np

from src.gnss_solver import solve_position_least_squares
from src.pseudorange_correlogram import (
    generate_synthetic_correlogram_case,
    parse_urbannav_ground_truth_text,
    pseudorange_correlogram_score,
    solve_correlogram_time_series,
    solve_position_pseudorange_correlogram,
    summarize_position_errors,
)
from src.simulation import generate_satellite_positions


class UrbanNavGroundTruthParserTests(unittest.TestCase):
    def test_parser_converts_dms_lat_lon_to_local_meter_positions(self):
        text = """      UTCTime       Week   GPSTime         Latitude        Longitude        H-Ell VelBdyX VelBdyY VelBdyZ AccBdyX AccBdyY AccBdyZ           Roll          Pitch        Heading Q
        (sec)    (weeks)     (sec)       (+/-D M S)       (+/-D M S)          (m)   (m/s)   (m/s)   (m/s) (m/s^2) (m/s^2) (m/s^2)          (deg)          (deg)          (deg)
1621218775.00 2158.00000  95593.00   22 18 04.00000  114 10 44.00000        3.472  -0.000   0.002  -0.010   0.381  -0.040   0.019  -1.7398149928   0.4409540487 -132.534729738 2
1621218776.00 2158.00000  95594.00   22 18 05.00000  114 10 45.00000        3.471  -0.000  -0.000  -0.011   0.127   0.144   0.088  -1.7380961194   0.4408681050 -132.534460448 2
"""

        positions = parse_urbannav_ground_truth_text(text)

        self.assertEqual(positions.shape, (2, 2))
        np.testing.assert_allclose(positions[0], np.array([0.0, 0.0]), atol=1e-6)
        self.assertGreater(positions[1, 0], 20.0)
        self.assertGreater(positions[1, 1], 25.0)


class PseudorangeCorrelogramTests(unittest.TestCase):
    def test_score_prefers_candidate_consistent_with_clean_pseudoranges(self):
        receiver = np.array([100.0, -50.0])
        clock_bias = 38.0
        satellites = generate_satellite_positions(num_satellites=7)
        pseudoranges = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        cn0 = np.full(len(pseudoranges), 45.0)

        true_score = pseudorange_correlogram_score(receiver, satellites, pseudoranges, cn0, clock_bias)
        offset_score = pseudorange_correlogram_score(
            receiver + np.array([450.0, 350.0]),
            satellites,
            pseudoranges,
            cn0,
            clock_bias,
        )

        self.assertGreater(true_score, offset_score)

    def test_grid_solution_is_less_pulled_by_low_cn0_nlos_than_ols(self):
        receiver = np.array([250.0, -120.0])
        clock_bias = 42.0
        satellites = generate_satellite_positions(num_satellites=8, angular_offset=0.15)
        clean = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        biased = clean.copy()
        biased[[0, 1]] += np.array([340.0, 290.0])
        cn0 = np.array([18.0, 20.0, 45.0, 43.0, 46.0, 44.0, 42.0, 45.0])

        ols = solve_position_least_squares(biased, satellites)
        correlogram = solve_position_pseudorange_correlogram(
            biased,
            satellites,
            cn0,
            initial_position=ols.position,
            grid_radius_m=260.0,
            grid_step_m=10.0,
        )

        ols_error = np.linalg.norm(ols.position - receiver)
        correlogram_error = np.linalg.norm(correlogram.position - receiver)

        self.assertLess(correlogram_error, ols_error)
        self.assertLess(correlogram_error, 35.0)
        self.assertTrue(np.isfinite(correlogram.score))

    def test_clock_bias_search_returns_finite_3d_score_grid(self):
        receiver = np.array([180.0, 75.0])
        clock_bias = 52.0
        satellites = generate_satellite_positions(num_satellites=7, angular_offset=0.22)
        pseudoranges = np.linalg.norm(satellites - receiver, axis=1) + clock_bias
        cn0 = np.full(len(pseudoranges), 45.0)

        solution = solve_position_pseudorange_correlogram(
            pseudoranges,
            satellites,
            cn0,
            initial_position=receiver + np.array([18.0, -12.0]),
            grid_radius_m=30.0,
            grid_step_m=10.0,
            clock_bias_search_m=np.arange(32.0, 73.0, 10.0),
        )

        self.assertEqual(solution.full_score_grid.shape, (5, 7, 7))
        self.assertEqual(solution.score_grid.shape, (7, 7))
        self.assertTrue(np.isfinite(solution.score))
        self.assertAlmostEqual(solution.clock_bias, clock_bias, delta=12.0)

    def test_clean_scenario_behaves_similarly_to_ols(self):
        route = np.column_stack((np.linspace(0.0, 120.0, 24), np.linspace(0.0, 40.0, 24)))
        case = generate_synthetic_correlogram_case(route, seed=33, scenario="clean_open_sky")

        ols_positions, ols_clock, _, _ = solve_correlogram_reference_ols(case)
        correlogram = solve_correlogram_time_series(
            case["pseudoranges"],
            case["satellite_positions"],
            case["cn0"],
            initial_positions=ols_positions,
            initial_clock_biases=ols_clock,
            grid_radius_m=30.0,
            grid_step_m=10.0,
            clock_bias_radius_m=20.0,
            clock_bias_step_m=10.0,
        )

        ols_mean = summarize_position_errors(case["true_positions"], ols_positions)["mean_error_m"]
        corr_mean = summarize_position_errors(case["true_positions"], correlogram["positions"])["mean_error_m"]

        self.assertLess(abs(corr_mean - ols_mean), 8.0)

    def test_nlos_scenario_shows_reasonable_robustness_improvement(self):
        route = np.column_stack((np.linspace(0.0, 160.0, 30), 35.0 * np.sin(np.linspace(0.0, np.pi, 30))))
        case = generate_synthetic_correlogram_case(route, seed=44, scenario="nlos_urban")

        ols_positions, ols_clock, _, _ = solve_correlogram_reference_ols(case)
        correlogram = solve_correlogram_time_series(
            case["pseudoranges"],
            case["satellite_positions"],
            case["cn0"],
            initial_positions=ols_positions,
            initial_clock_biases=ols_clock,
            grid_radius_m=240.0,
            grid_step_m=20.0,
            clock_bias_radius_m=80.0,
            clock_bias_step_m=20.0,
        )

        ols_mean = summarize_position_errors(case["true_positions"], ols_positions)["mean_error_m"]
        corr_mean = summarize_position_errors(case["true_positions"], correlogram["positions"])["mean_error_m"]

        self.assertLess(corr_mean, ols_mean * 0.92)


def solve_correlogram_reference_ols(case):
    from src.gnss_solver import solve_time_series

    return solve_time_series(
        case["pseudoranges"],
        case["satellite_positions"],
        initial_position=case["true_positions"][0],
        method="ols",
    )


if __name__ == "__main__":
    unittest.main()
