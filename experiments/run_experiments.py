"""Run reproducible MiniLab experiments and save CSV/figures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_data import (
    clean_leo_comparison_results,
    collaborative_results,
    gnss_ins_results,
    robust_solver_results,
    sliding_window_results,
)
from src.simulation import run_comparison
from src.urban_environment import urban_error_map_results, urban_los_nlos_demo_results


RESULTS_DIR = PROJECT_ROOT / "results"


def _metrics_from_errors(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean_positioning_error_m": float(np.mean(errors)),
        "median_positioning_error_m": float(np.median(errors)),
        "max_positioning_error_m": float(np.max(errors)),
        "rmse_positioning_error_m": float(np.sqrt(np.mean(errors**2))),
        "p95_positioning_error_m": float(np.percentile(errors, 95.0)),
    }


def collect_summary_metrics() -> pd.DataFrame:
    """Collect fixed metrics from scenario, robust-LS, and GNSS/INS studies."""

    rows: list[dict[str, float | str]] = []

    for row in run_comparison(seed=4):
        rows.append(
            {
                "experiment_family": "scenario_comparison",
                "case": str(row["scenario"]),
                "mean_positioning_error_m": float(row["mean_error_m"]),
                "median_positioning_error_m": float(row["median_error_m"]),
                "max_positioning_error_m": float(row["max_error_m"]),
                "rmse_positioning_error_m": float(row["rmse_error_m"]),
                "p95_positioning_error_m": float(row["p95_error_m"]),
                "mean_hdop": float(row["mean_hdop"]),
                "max_hdop": float(row["max_hdop"]),
                "anomaly_count": float(row["anomaly_count"]),
            }
        )

    robust = robust_solver_results()
    true_positions = robust["true_positions"]
    for method, positions in robust["solutions"].items():
        errors = np.linalg.norm(positions - true_positions, axis=1)
        row = {
            "experiment_family": "robust_solver_comparison",
            "case": str(method),
            **_metrics_from_errors(errors),
            "mean_hdop": float(np.mean(robust["hdop"])),
            "max_hdop": float(np.max(robust["hdop"])),
            "anomaly_count": float(np.sum(robust["nlos_mask"])),
        }
        rows.append(row)

    clean = clean_leo_comparison_results()
    for case_key, case_label in [
        ("gnss_only", "Clean GNSS-only"),
        ("gnss_leo", "Clean GNSS + LEO-like"),
    ]:
        rows.append(
            {
                "experiment_family": "clean_leo_comparison",
                "case": case_label,
                **_metrics_from_errors(clean[case_key]["errors"]),
                "mean_hdop": float(np.mean(clean[case_key]["hdop"])),
                "max_hdop": float(np.max(clean[case_key]["hdop"])),
                "anomaly_count": 0.0,
            }
        )

    ins = gnss_ins_results()
    true_positions = ins["true_positions"]
    for case, positions in [
        ("GNSS-only degraded", ins["gnss_positions"]),
        ("INS/odometry-only", ins["ins_only"]),
        ("GNSS/INS toy KF", ins["kf_positions"]),
    ]:
        errors = np.linalg.norm(positions - true_positions, axis=1)
        rows.append(
            {
                "experiment_family": "gnss_ins_comparison",
                "case": case,
                **_metrics_from_errors(errors),
                "mean_hdop": np.nan,
                "max_hdop": np.nan,
                "anomaly_count": np.nan,
            }
        )

    urban = urban_error_map_results(grid_size=21, seed=22)
    for case, grid, mean_key, rmse_key in [
        ("OLS urban grid", urban["ols_error_grid"], "ols_mean_error_m", "ols_rmse_error_m"),
        ("WLS urban grid", urban["wls_error_grid"], "wls_mean_error_m", "wls_rmse_error_m"),
    ]:
        rows.append(
            {
                "experiment_family": "urban_error_map",
                "case": case,
                "mean_positioning_error_m": float(urban["metrics"][mean_key]),
                "median_positioning_error_m": float(np.median(grid)),
                "max_positioning_error_m": float(np.max(grid)),
                "rmse_positioning_error_m": float(urban["metrics"][rmse_key]),
                "p95_positioning_error_m": float(np.percentile(grid, 95.0)),
                "mean_hdop": float(urban["metrics"]["mean_hdop"]),
                "max_hdop": float(urban["metrics"]["max_hdop"]),
                "anomaly_count": float(urban["metrics"]["mean_nlos_count"]),
            }
        )

    sliding = sliding_window_results(seed=52)
    for case, errors, mean_key in [
        ("Single-epoch OLS", sliding["single_epoch_errors"], "single_epoch_mean_error_m"),
        ("Sliding-window LS", sliding["sliding_errors"], "sliding_window_mean_error_m"),
    ]:
        rows.append(
            {
                "experiment_family": "sliding_window_optimization",
                "case": case,
                "mean_positioning_error_m": float(sliding["metrics"][mean_key]),
                "median_positioning_error_m": float(np.median(errors)),
                "max_positioning_error_m": float(np.max(errors)),
                "rmse_positioning_error_m": float(np.sqrt(np.mean(errors**2))),
                "p95_positioning_error_m": float(np.percentile(errors, 95.0)),
                "mean_hdop": np.nan,
                "max_hdop": np.nan,
                "anomaly_count": np.nan,
            }
        )

    collaborative = collaborative_results(seed=61)
    for case, errors, mean_key in [
        ("Agent B GNSS-only", collaborative["gnss_errors_b"], "gnss_only_mean_error_b_m"),
        (
            "Agent B collaborative",
            collaborative["collaborative_errors_b"],
            "collaborative_mean_error_b_m",
        ),
    ]:
        rows.append(
            {
                "experiment_family": "collaborative_positioning",
                "case": case,
                "mean_positioning_error_m": float(collaborative["metrics"][mean_key]),
                "median_positioning_error_m": float(np.median(errors)),
                "max_positioning_error_m": float(np.max(errors)),
                "rmse_positioning_error_m": float(np.sqrt(np.mean(errors**2))),
                "p95_positioning_error_m": float(np.percentile(errors, 95.0)),
                "mean_hdop": np.nan,
                "max_hdop": np.nan,
                "anomaly_count": np.nan,
            }
        )

    return pd.DataFrame(rows)


def save_urban_tables(results_dir: Path | str = RESULTS_DIR) -> list[Path]:
    """Save CSV tables for the urban LOS/NLOS and error-map experiments."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urban = urban_error_map_results(grid_size=21, seed=22)
    rows = []
    for row_index, y_value in enumerate(urban["y_grid"]):
        for col_index, x_value in enumerate(urban["x_grid"]):
            rows.append(
                {
                    "east_m": float(x_value),
                    "north_m": float(y_value),
                    "ols_error_m": float(urban["ols_error_grid"][row_index, col_index]),
                    "wls_error_m": float(urban["wls_error_grid"][row_index, col_index]),
                    "hdop": float(urban["hdop_grid"][row_index, col_index]),
                    "nlos_measurement_count": float(urban["nlos_count_grid"][row_index, col_index]),
                }
            )
    error_map_path = output_dir / "urban_error_map.csv"
    pd.DataFrame(rows).to_csv(error_map_path, index=False)

    labels = urban_los_nlos_demo_results(num_steps=80, seed=22)
    label_rows = []
    nlos_mask = labels["nlos_mask"]
    sigmas = labels["measurement_sigmas"]
    for epoch in range(nlos_mask.shape[0]):
        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "east_m": float(labels["trajectory"][epoch, 0]),
            "north_m": float(labels["trajectory"][epoch, 1]),
            "nlos_measurement_count": int(np.sum(nlos_mask[epoch])),
        }
        for source_index in range(nlos_mask.shape[1]):
            row[f"source_{source_index}_is_nlos"] = int(nlos_mask[epoch, source_index])
            row[f"source_{source_index}_sigma_m"] = float(sigmas[epoch, source_index])
        label_rows.append(row)
    labels_path = output_dir / "urban_los_nlos_labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)

    return [error_map_path, labels_path]


def save_core_research_tables(results_dir: Path | str = RESULTS_DIR) -> list[Path]:
    """Save CSV tables for solver, GNSS/INS, and LEO-like aiding studies."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    robust = robust_solver_results()
    true_positions = robust["true_positions"]
    solver_rows = []
    for method, positions in robust["solutions"].items():
        errors = np.linalg.norm(positions - true_positions, axis=1)
        solver_rows.append(
            {
                "method": method,
                "mean_error_m": float(np.mean(errors)),
                "median_error_m": float(np.median(errors)),
                "rmse_error_m": float(np.sqrt(np.mean(errors**2))),
                "max_error_m": float(np.max(errors)),
                "nlos_measurement_count": int(np.sum(robust["nlos_mask"])),
            }
        )
    solver_path = output_dir / "solver_comparison_metrics.csv"
    pd.DataFrame(solver_rows).to_csv(solver_path, index=False)
    paths.append(solver_path)

    ins = gnss_ins_results()
    true_positions = ins["true_positions"]
    ins_rows = []
    for case, positions in [
        ("GNSS-only degraded", ins["gnss_positions"]),
        ("INS/odometry-only", ins["ins_only"]),
        ("GNSS/INS toy KF", ins["kf_positions"]),
    ]:
        errors = np.linalg.norm(positions - true_positions, axis=1)
        ins_rows.append(
            {
                "case": case,
                "mean_error_m": float(np.mean(errors)),
                "median_error_m": float(np.median(errors)),
                "rmse_error_m": float(np.sqrt(np.mean(errors**2))),
                "max_error_m": float(np.max(errors)),
            }
        )
    ins_path = output_dir / "gnss_ins_kf_metrics.csv"
    pd.DataFrame(ins_rows).to_csv(ins_path, index=False)
    paths.append(ins_path)

    clean = clean_leo_comparison_results()
    degraded_gnss = run_comparison(seed=18, measurement_noise_std=8.0, multipath_bias_level=80.0)
    degraded_lookup = {row["scenario"]: row for row in degraded_gnss}
    leo_rows = []
    for case, payload in [
        ("Clean GNSS-only", clean["gnss_only"]),
        ("Clean GNSS + LEO-like", clean["gnss_leo"]),
    ]:
        errors = payload["errors"]
        hdop = payload["hdop"]
        leo_rows.append(
            {
                "case": case,
                "mean_error_m": float(np.mean(errors)),
                "rmse_error_m": float(np.sqrt(np.mean(errors**2))),
                "mean_hdop": float(np.mean(hdop)),
                "max_hdop": float(np.max(hdop)),
            }
        )
    for scenario in ["Multipath/NLOS", "GNSS + LEO-like aiding"]:
        row = degraded_lookup[scenario]
        leo_rows.append(
            {
                "case": f"Degraded {scenario}",
                "mean_error_m": float(row["mean_error_m"]),
                "rmse_error_m": float(row["rmse_error_m"]),
                "mean_hdop": float(row["mean_hdop"]),
                "max_hdop": float(row["max_hdop"]),
            }
        )
    leo_path = output_dir / "leo_aiding_metrics.csv"
    pd.DataFrame(leo_rows).to_csv(leo_path, index=False)
    paths.append(leo_path)

    return paths


def save_advanced_research_tables(results_dir: Path | str = RESULTS_DIR) -> list[Path]:
    """Save CSV metrics for sliding-window and collaborative-positioning demos."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sliding = sliding_window_results(seed=52)
    sliding_path = output_dir / "sliding_window_metrics.csv"
    pd.DataFrame(
        [
            {
                "case": "Single-epoch OLS",
                "mean_error_m": float(sliding["metrics"]["single_epoch_mean_error_m"]),
                "degraded_interval_mean_error_m": float(sliding["metrics"]["single_epoch_degraded_mean_error_m"]),
                "rmse_error_m": float(np.sqrt(np.mean(sliding["single_epoch_errors"] ** 2))),
            },
            {
                "case": "Sliding-window LS",
                "mean_error_m": float(sliding["metrics"]["sliding_window_mean_error_m"]),
                "degraded_interval_mean_error_m": float(sliding["metrics"]["sliding_window_degraded_mean_error_m"]),
                "rmse_error_m": float(sliding["metrics"]["sliding_window_rmse_error_m"]),
            },
        ]
    ).to_csv(sliding_path, index=False)

    collaborative = collaborative_results(seed=61)
    collaborative_path = output_dir / "collaborative_positioning_metrics.csv"
    pd.DataFrame(
        [
            {
                "case": "Agent B GNSS-only",
                "mean_error_m": float(collaborative["metrics"]["gnss_only_mean_error_b_m"]),
                "rmse_error_m": float(collaborative["metrics"]["gnss_only_rmse_b_m"]),
            },
            {
                "case": "Agent B collaborative",
                "mean_error_m": float(collaborative["metrics"]["collaborative_mean_error_b_m"]),
                "rmse_error_m": float(collaborative["metrics"]["collaborative_rmse_b_m"]),
            },
        ]
    ).to_csv(collaborative_path, index=False)

    return [sliding_path, collaborative_path]


def main() -> None:
    from experiments.plot_results import generate_all_plots

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = collect_summary_metrics()
    summary_path = RESULTS_DIR / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(summary_path)
    for table_path in save_core_research_tables(RESULTS_DIR):
        print(table_path)
    for table_path in save_urban_tables(RESULTS_DIR):
        print(table_path)
    for table_path in save_advanced_research_tables(RESULTS_DIR):
        print(table_path)
    for figure_path in generate_all_plots(RESULTS_DIR):
        print(figure_path)


if __name__ == "__main__":
    main()
