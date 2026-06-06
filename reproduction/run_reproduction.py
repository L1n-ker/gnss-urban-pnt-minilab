"""Run the literature-grounded pseudorange-correlogram reproduction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reproduction.plot_reproduction import generate_reproduction_plots
from src.gnss_solver import solve_time_series
from src.pseudorange_correlogram import (
    ensure_urbannav_ground_truth_file,
    generate_synthetic_correlogram_case,
    load_urbannav_ground_truth_file,
    solve_correlogram_time_series,
    summarize_position_errors,
)


DATA_DIR = PROJECT_ROOT / "reproduction" / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "reproduction"
URBANNAV_GT_FILE = DATA_DIR / "UrbanNav_TST_GT_raw.txt"


def run_reproduction() -> dict[str, Path]:
    """Run the selected-paper-inspired scenarios and save all artifacts."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_urbannav_ground_truth_file(URBANNAV_GT_FILE)
    route_positions = load_urbannav_ground_truth_file(URBANNAV_GT_FILE, max_points=90)
    scenario_runs = [
        _run_single_scenario(route_positions, "clean_open_sky", seed=2026),
        _run_single_scenario(route_positions, "nlos_urban", seed=2027),
    ]

    metrics = pd.DataFrame(row for run in scenario_runs for row in run["metric_rows"])
    scenario_metrics_path = RESULTS_DIR / "reproduction_scenario_metrics.csv"
    metrics.to_csv(scenario_metrics_path, index=False)

    metrics_path = RESULTS_DIR / "reproduction_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    epoch_errors = pd.concat([run["epoch_errors"] for run in scenario_runs], ignore_index=True)
    epoch_errors_path = RESULTS_DIR / "reproduction_epoch_errors.csv"
    epoch_errors.to_csv(epoch_errors_path, index=False)

    clean = scenario_runs[0]
    nlos = scenario_runs[1]
    selected_solution = nlos["selected_solution"]
    selected_epoch = int(nlos["selected_epoch"])

    arrays_path = RESULTS_DIR / "correlogram_reproduction_arrays.npz"
    np.savez(
        arrays_path,
        clean_true_positions=clean["case"]["true_positions"],
        clean_ols_positions=clean["ols_positions"],
        clean_correlogram_positions=clean["correlogram_positions"],
        clean_ols_errors=clean["ols_errors"],
        clean_correlogram_errors=clean["correlogram_errors"],
        clean_nlos_counts=np.sum(clean["case"]["nlos_mask"], axis=1),
        nlos_true_positions=nlos["case"]["true_positions"],
        nlos_ols_positions=nlos["ols_positions"],
        nlos_correlogram_positions=nlos["correlogram_positions"],
        nlos_ols_errors=nlos["ols_errors"],
        nlos_correlogram_errors=nlos["correlogram_errors"],
        nlos_nlos_counts=np.sum(nlos["case"]["nlos_mask"], axis=1),
        nlos_multipath_counts=np.sum(nlos["case"]["multipath_mask"], axis=1),
        nlos_scores=nlos["correlogram"]["scores"],
        selected_epoch=np.array(selected_epoch),
        score_grid=selected_solution.score_grid,
        full_score_grid=selected_solution.full_score_grid,
        x_grid=selected_solution.x_grid,
        y_grid=selected_solution.y_grid,
        clock_bias_grid=selected_solution.clock_bias_grid,
        best_clock_bias_m=np.array(selected_solution.clock_bias),
        true_positions=nlos["case"]["true_positions"],
        ols_positions=nlos["ols_positions"],
        correlogram_positions=nlos["correlogram_positions"],
        ols_errors=nlos["ols_errors"],
        correlogram_errors=nlos["correlogram_errors"],
        nlos_counts=np.sum(nlos["case"]["nlos_mask"], axis=1),
        scores=nlos["correlogram"]["scores"],
    )

    figure_paths = generate_reproduction_plots(RESULTS_DIR)
    notes_path = RESULTS_DIR / "reproduction_notes.md"
    notes_path.write_text(
        _build_notes(metrics, selected_epoch, float(selected_solution.clock_bias), figure_paths),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_path,
        "scenario_metrics": scenario_metrics_path,
        "epoch_errors": epoch_errors_path,
        "arrays": arrays_path,
        "notes": notes_path,
        **{f"figure_{index + 1}": path for index, path in enumerate(figure_paths)},
    }


def _run_single_scenario(route_positions: np.ndarray, scenario: str, seed: int) -> dict[str, object]:
    case = generate_synthetic_correlogram_case(route_positions, seed=seed, num_satellites=8, scenario=scenario)
    ols_positions, ols_clock, _, _ = solve_time_series(
        case["pseudoranges"],
        case["satellite_positions"],
        initial_position=case["true_positions"][0],
        method="ols",
    )

    if scenario == "clean_open_sky":
        grid_radius_m = 40.0
        grid_step_m = 20.0
        clock_bias_radius_m = 20.0
        clock_bias_step_m = 20.0
        selected_epoch = len(route_positions) // 2
        scenario_label = "clean/open-sky-like synthetic setup"
    else:
        grid_radius_m = 240.0
        grid_step_m = 20.0
        clock_bias_radius_m = 80.0
        clock_bias_step_m = 20.0
        selected_epoch = int(np.argmax(np.sum(case["nlos_mask"], axis=1)))
        scenario_label = "NLOS-dominated urban synthetic setup"

    correlogram = solve_correlogram_time_series(
        case["pseudoranges"],
        case["satellite_positions"],
        case["cn0"],
        initial_positions=ols_positions,
        initial_clock_biases=ols_clock,
        grid_radius_m=grid_radius_m,
        grid_step_m=grid_step_m,
        clock_bias_radius_m=clock_bias_radius_m,
        clock_bias_step_m=clock_bias_step_m,
        selected_epoch=selected_epoch,
    )
    correlogram_positions = correlogram["positions"]
    ols_errors = np.linalg.norm(ols_positions - case["true_positions"], axis=1)
    correlogram_errors = np.linalg.norm(correlogram_positions - case["true_positions"], axis=1)
    ols_metrics = summarize_position_errors(case["true_positions"], ols_positions)
    corr_metrics = summarize_position_errors(case["true_positions"], correlogram_positions)
    improvement_pct = 100.0 * (ols_metrics["mean_error_m"] - corr_metrics["mean_error_m"]) / max(
        ols_metrics["mean_error_m"],
        1e-9,
    )

    provenance = {
        "route_source": "public UrbanNav-HK-Medium-Urban-1 TST ground-truth route subset",
        "pseudorange_source": "synthetic pseudorange generated in this project",
        "cn0_source": "synthetic C/N0 generated in this project",
        "urban_nav_rinex_used": "no",
        "paper_table_values_used": "no",
        "paper_figure_values_digitized": "no",
    }
    metric_rows = [
        {
            "scenario": scenario,
            "scenario_label": scenario_label,
            "method": "OLS baseline",
            **ols_metrics,
            "mean_correlogram_score": np.nan,
            "mean_improvement_vs_ols_pct": np.nan,
            "mean_los_cn0_dbhz": float(np.mean(case["cn0"][~case["nlos_mask"] & ~case["multipath_mask"]])),
            "mean_multipath_cn0_dbhz": _safe_mask_mean(case["cn0"], case["multipath_mask"]),
            "mean_nlos_cn0_dbhz": _safe_mask_mean(case["cn0"], case["nlos_mask"]),
            **provenance,
        },
        {
            "scenario": scenario,
            "scenario_label": scenario_label,
            "method": "Pseudorange-correlogram-inspired toy implementation",
            **corr_metrics,
            "mean_correlogram_score": float(np.mean(correlogram["scores"])),
            "mean_improvement_vs_ols_pct": float(improvement_pct),
            "mean_los_cn0_dbhz": float(np.mean(case["cn0"][~case["nlos_mask"] & ~case["multipath_mask"]])),
            "mean_multipath_cn0_dbhz": _safe_mask_mean(case["cn0"], case["multipath_mask"]),
            "mean_nlos_cn0_dbhz": _safe_mask_mean(case["cn0"], case["nlos_mask"]),
            **provenance,
        },
    ]

    epoch_errors = pd.DataFrame(
        {
            "scenario": scenario,
            "epoch": np.arange(len(ols_errors)),
            "ols_error_m": ols_errors,
            "correlogram_error_m": correlogram_errors,
            "multipath_measurement_count": np.sum(case["multipath_mask"], axis=1),
            "nlos_measurement_count": np.sum(case["nlos_mask"], axis=1),
            "correlogram_score": correlogram["scores"],
        }
    )

    return {
        "case": case,
        "ols_positions": ols_positions,
        "ols_clock": ols_clock,
        "correlogram": correlogram,
        "correlogram_positions": correlogram_positions,
        "ols_errors": ols_errors,
        "correlogram_errors": correlogram_errors,
        "metric_rows": metric_rows,
        "epoch_errors": epoch_errors,
        "selected_epoch": selected_epoch,
        "selected_solution": correlogram["selected_solution"],
    }


def _safe_mask_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def _build_notes(metrics: pd.DataFrame, selected_epoch: int, best_clock_bias_m: float, figure_paths: list[Path]) -> str:
    clean = metrics[metrics["scenario"] == "clean_open_sky"]
    nlos = metrics[metrics["scenario"] == "nlos_urban"]
    clean_ols = clean[clean["method"] == "OLS baseline"].iloc[0]
    clean_corr = clean[clean["method"] != "OLS baseline"].iloc[0]
    nlos_ols = nlos[nlos["method"] == "OLS baseline"].iloc[0]
    nlos_corr = nlos[nlos["method"] != "OLS baseline"].iloc[0]
    return f"""# Reproduction Notes

## What Was Reproduced

This run is a simplified reproduction / conceptual replication of one component inspired by Vicenzo et al. (2024): pseudorange-correlogram candidate-state scoring with C/N0 weighting. It compares the MiniLab's ordinary least-squares baseline with a pseudorange-correlogram-inspired toy implementation under two synthetic-measurement scenarios.

## Data Source

- Public data used: UrbanNav-HK-Medium-Urban-1 TST ground-truth route subset, downloaded from the public UrbanNav dataset links.
- Synthetic data used: satellite positions, pseudorange measurements, receiver clock bias in meters, LOS/multipath/NLOS biases, and C/N0 values.
- No UrbanNav RINEX pseudorange or real C/N0 is parsed by this project.
- No paper table values are manually transcribed.
- No values are digitized from paper figures.
- This is not original raw data from the selected paper and should not be described as such.

## Scenario Metrics

| Scenario | Method | Mean error (m) | RMSE (m) | Max error (m) | Improvement vs OLS |
|---|---|---:|---:|---:|---:|
| Clean/open-sky-like | OLS baseline | {clean_ols['mean_error_m']:.3f} | {clean_ols['rmse_error_m']:.3f} | {clean_ols['max_error_m']:.3f} | n/a |
| Clean/open-sky-like | Correlogram toy | {clean_corr['mean_error_m']:.3f} | {clean_corr['rmse_error_m']:.3f} | {clean_corr['max_error_m']:.3f} | {clean_corr['mean_improvement_vs_ols_pct']:.2f}% |
| NLOS-dominated urban | OLS baseline | {nlos_ols['mean_error_m']:.3f} | {nlos_ols['rmse_error_m']:.3f} | {nlos_ols['max_error_m']:.3f} | n/a |
| NLOS-dominated urban | Correlogram toy | {nlos_corr['mean_error_m']:.3f} | {nlos_corr['rmse_error_m']:.3f} | {nlos_corr['max_error_m']:.3f} | {nlos_corr['mean_improvement_vs_ols_pct']:.2f}% |

Expected qualitative behavior is preserved: OLS and the correlogram toy are similar in the clean/open-sky-like setup, while the correlogram toy is more robust in the NLOS-dominated synthetic setup.

## Generated Figures

{chr(10).join(f'- `{path.as_posix()}`' for path in figure_paths)}

## Approximation Boundary

The XY score-slice figure uses epoch {selected_epoch} from the synthetic NLOS-dominated scenario and is shown at the best searched clock-bias value, {best_clock_bias_m:.3f} m. Clock bias is expressed in meters in this simplified pseudorange model.

## Differences From The Original Paper

- The paper evaluates real UrbanNav/RINEX-derived positioning and simulation data; this project uses a public UrbanNav route with synthetic measurements.
- The paper is a 3D GNSS urban-navigation workflow; this project is a 2D educational simulator.
- The paper's generated results are available from the corresponding author on request; this project does not claim to reproduce those raw result files or the exact original UrbanNav/RINEX experiment.

## Future Extensions

- Parse UrbanNav RINEX pseudorange/C/N0 and skymask labels.
- Replace fixed synthetic satellite geometry with real ephemeris-derived geometry.
- Compare against WLS/Huber/residual-exclusion baselines in the same reproduction script.
- Reproduce one reported table only if values are manually transcribed and labeled as reported values from the paper.
"""


def main() -> None:
    for label, path in run_reproduction().items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
