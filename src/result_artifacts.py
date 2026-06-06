"""Helpers for loading generated result artifacts in the desktop GUI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FIGURE_FILENAMES = [
    "scenario_comparison.png",
    "dop_vs_error.png",
    "clean_leo_hdop_comparison.png",
    "clean_leo_error_comparison.png",
    "robust_ls_comparison.png",
    "nlos_weighting_demo.png",
    "gnss_ins_comparison.png",
    "solver_comparison_controlled.png",
    "nlos_uncertainty_weighting.png",
    "gnss_ins_kf_demo.png",
    "leo_clean_geometry_comparison.png",
    "leo_degraded_aiding_comparison.png",
    "urban_los_nlos_demo.png",
    "urban_error_map.png",
    "urban_hdop_map.png",
    "sliding_window_demo.png",
    "collaborative_positioning_demo.png",
    "collaborative_positioning_error.png",
    "reproduction/reproduced_figure_error_timeseries.png",
    "reproduction/reproduced_figure_trajectory.png",
    "reproduction/reproduced_figure_score_map.png",
    "reproduction/correlogram_xy_slice.png",
    "reproduction/reproduction_clean_vs_nlos.png",
]


def project_root_from_here() -> Path:
    """Return the project root based on this helper module's location."""

    return Path(__file__).resolve().parents[1]


def get_result_path(filename: str, project_root: Path | str | None = None) -> Path:
    """Return the absolute path to a file under ``results/``."""

    root = Path(project_root) if project_root is not None else project_root_from_here()
    return root / "results" / filename


def result_file_exists(filename: str, project_root: Path | str | None = None) -> bool:
    """Check whether a generated result artifact exists."""

    return get_result_path(filename, project_root).is_file()


def load_summary_metrics(project_root: Path | str | None = None) -> pd.DataFrame:
    """Load ``results/summary_metrics.csv``."""

    path = get_result_path("summary_metrics.csv", project_root)
    if not path.is_file():
        raise FileNotFoundError(missing_results_message())
    return pd.read_csv(path)


def missing_results_message() -> str:
    """Message shown when generated artifacts are missing."""

    return "Please run `python experiments/run_all_experiments.py` first."
