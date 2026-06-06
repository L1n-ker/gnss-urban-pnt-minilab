import unittest

from src.desktop_config import (
    CHINESE_SCENARIOS,
    METRIC_LABELS,
    build_metric_rows,
    scenario_to_internal,
)


class DesktopConfigTests(unittest.TestCase):
    def test_chinese_scenarios_map_to_internal_names(self):
        self.assertEqual(scenario_to_internal("理想 GNSS"), "Ideal GNSS")
        self.assertEqual(scenario_to_internal("GNSS + INS + LEO 类辅助"), "GNSS + INS + LEO-like aiding")

    def test_all_scenarios_have_chinese_labels(self):
        self.assertGreaterEqual(len(CHINESE_SCENARIOS), 6)
        self.assertTrue(all(label for label in CHINESE_SCENARIOS))
        self.assertIn("多路径/NLOS", CHINESE_SCENARIOS)

    def test_metric_rows_are_chinese_and_ordered(self):
        metrics = {
            "mean_error_m": 1.234,
            "median_error_m": 1.0,
            "max_error_m": 5.5,
            "final_error_m": 2.25,
            "mean_residual_rms_m": 0.8,
            "max_residual_rms_m": 3.6,
            "anomaly_count": 2.0,
            "convergence_rate_pct": 100.0,
        }

        rows = build_metric_rows(metrics)

        self.assertEqual(rows[0][0], METRIC_LABELS["mean_error_m"])
        self.assertEqual(rows[0][1], "1.23")
        self.assertEqual(rows[-1][0], "收敛率 (%)")
        self.assertEqual(rows[-1][1], "100.00")


if __name__ == "__main__":
    unittest.main()
