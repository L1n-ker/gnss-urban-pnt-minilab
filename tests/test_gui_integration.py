import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.experiment_data import clean_leo_comparison_results
from src.result_artifacts import (
    FIGURE_FILENAMES,
    get_result_path,
    load_summary_metrics,
    missing_results_message,
    result_file_exists,
)


class ResultArtifactTests(unittest.TestCase):
    def test_result_paths_resolve_under_results_directory(self):
        path = get_result_path("dop_vs_error.png")

        self.assertEqual(path.name, "dop_vs_error.png")
        self.assertEqual(path.parent.name, "results")

    def test_missing_results_message_names_command(self):
        message = missing_results_message()

        self.assertIn("python experiments/run_all_experiments.py", message)

    def test_required_pipeline_scripts_exist(self):
        project_root = Path(__file__).resolve().parents[1]

        self.assertTrue((project_root / "experiments" / "run_all_experiments.py").is_file())
        self.assertTrue((project_root / "experiments" / "plot_all_results.py").is_file())
        self.assertTrue((project_root / "experiments" / "make_summary_tables.py").is_file())

    def test_result_file_exists_accepts_custom_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            (results / "example.png").write_bytes(b"fake")

            self.assertTrue(result_file_exists("example.png", project_root=root))
            self.assertFalse(result_file_exists("missing.png", project_root=root))

    def test_load_summary_metrics_reads_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            expected = pd.DataFrame({"experiment_family": ["demo"], "mean_positioning_error_m": [1.2]})
            expected.to_csv(results / "summary_metrics.csv", index=False)

            loaded = load_summary_metrics(project_root=root)

            self.assertEqual(list(loaded.columns), list(expected.columns))
            self.assertAlmostEqual(float(loaded.iloc[0]["mean_positioning_error_m"]), 1.2)

    def test_required_figures_include_new_clean_leo_outputs(self):
        self.assertIn("clean_leo_hdop_comparison.png", FIGURE_FILENAMES)
        self.assertIn("clean_leo_error_comparison.png", FIGURE_FILENAMES)

    def test_required_figures_include_reproduction_and_urban_outputs(self):
        self.assertIn("urban_error_map.png", FIGURE_FILENAMES)
        self.assertIn("urban_los_nlos_demo.png", FIGURE_FILENAMES)
        self.assertIn("reproduction/reproduced_figure_error_timeseries.png", FIGURE_FILENAMES)
        self.assertIn("reproduction/reproduced_figure_trajectory.png", FIGURE_FILENAMES)
        self.assertIn("reproduction/reproduced_figure_score_map.png", FIGURE_FILENAMES)
        self.assertIn("sliding_window_demo.png", FIGURE_FILENAMES)
        self.assertIn("collaborative_positioning_demo.png", FIGURE_FILENAMES)
        self.assertIn("collaborative_positioning_error.png", FIGURE_FILENAMES)
        self.assertIn("solver_comparison_controlled.png", FIGURE_FILENAMES)
        self.assertIn("nlos_uncertainty_weighting.png", FIGURE_FILENAMES)
        self.assertIn("gnss_ins_kf_demo.png", FIGURE_FILENAMES)
        self.assertIn("leo_clean_geometry_comparison.png", FIGURE_FILENAMES)
        self.assertIn("leo_degraded_aiding_comparison.png", FIGURE_FILENAMES)


class CleanLeoComparisonTests(unittest.TestCase):
    def test_clean_leo_comparison_has_matching_error_and_hdop_series(self):
        result = clean_leo_comparison_results(seed=14)

        self.assertIn("gnss_only", result)
        self.assertIn("gnss_leo", result)
        for key in ["gnss_only", "gnss_leo"]:
            self.assertEqual(result[key]["errors"].shape, result[key]["hdop"].shape)
            self.assertTrue(np.all(np.isfinite(result[key]["errors"])))
            self.assertTrue(np.all(np.isfinite(result[key]["hdop"])))

    def test_clean_leo_comparison_improves_mean_hdop(self):
        result = clean_leo_comparison_results(seed=14)

        self.assertLess(float(np.mean(result["gnss_leo"]["hdop"])), float(np.mean(result["gnss_only"]["hdop"])))


if __name__ == "__main__":
    unittest.main()
