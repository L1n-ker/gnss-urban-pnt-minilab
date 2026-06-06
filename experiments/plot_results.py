"""Generate fixed, presentation-friendly plots for the MiniLab."""

from __future__ import annotations

import sys
from shutil import copyfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
from src.simulation import run_comparison, run_scenario
from src.urban_environment import UrbanBlock, urban_error_map_results, urban_los_nlos_demo_results


def _results_dir(results_dir: Path | str | None = None) -> Path:
    path = Path(results_dir) if results_dir is not None else PROJECT_ROOT / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_scenario_comparison(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "scenario_comparison.png"
    rows = pd.DataFrame(run_comparison(seed=4))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    ax1.bar(x - 0.18, rows["mean_error_m"], width=0.36, label="Mean error (m)", color="#2a9d8f")
    ax1.set_ylabel("Mean positioning error (m)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rows["scenario"], rotation=25, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, rows["mean_hdop"], marker="o", color="#264653", label="Mean HDOP")
    ax2.set_ylabel("Mean HDOP")
    fig.suptitle("Scenario Comparison: Error and Geometry")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_dop_vs_error(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "dop_vs_error.png"
    gnss = run_scenario("Multipath/NLOS", seed=12, measurement_noise_std=8.0, multipath_bias_level=70.0)
    aided = run_scenario(
        "GNSS + LEO-like aiding",
        enable_leo=True,
        num_leo=3,
        seed=12,
        measurement_noise_std=8.0,
        multipath_bias_level=70.0,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(gnss["hdop"], gnss["position_errors"], label="GNSS only", alpha=0.75, color="#e76f51")
    ax.scatter(aided["hdop"], aided["position_errors"], label="GNSS + LEO-like", alpha=0.75, color="#2a9d8f")
    ax.set_xlabel("HDOP")
    ax.set_ylabel("Positioning error (m)")
    ax.set_title("HDOP vs Positioning Error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_clean_leo_hdop_comparison(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "clean_leo_hdop_comparison.png"
    data = clean_leo_comparison_results()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["gnss_only"]["hdop"], label="Clean GNSS-only", color="#e76f51", linewidth=1.8)
    ax.plot(data["gnss_leo"]["hdop"], label="Clean GNSS + LEO-like", color="#2a9d8f", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("HDOP")
    ax.set_title("Clean Geometry Comparison: GNSS-only vs GNSS + LEO-like")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_clean_leo_error_comparison(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "clean_leo_error_comparison.png"
    data = clean_leo_comparison_results()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["gnss_only"]["errors"], label="Clean GNSS-only", color="#e76f51", linewidth=1.8)
    ax.plot(data["gnss_leo"]["errors"], label="Clean GNSS + LEO-like", color="#2a9d8f", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Positioning error (m)")
    ax.set_title("Clean Error Comparison: GNSS-only vs GNSS + LEO-like")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_robust_ls_comparison(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "robust_ls_comparison.png"
    data = robust_solver_results()
    true_positions = data["true_positions"]
    solutions = data["solutions"]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {
        "ols": "OLS",
        "wls": "WLS (NLOS downweighted)",
        "huber": "Huber IRLS",
        "cauchy": "Cauchy IRLS",
        "residual_exclusion": "Residual exclusion",
    }
    colors = {
        "ols": "#e76f51",
        "wls": "#2a9d8f",
        "huber": "#264653",
        "cauchy": "#0ea5e9",
        "residual_exclusion": "#8a5cf6",
    }
    for key, positions in solutions.items():
        errors = np.linalg.norm(positions - true_positions, axis=1)
        ax.plot(errors, label=labels[key], color=colors[key], linewidth=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Positioning error (m)")
    ax.set_title("OLS, WLS, Robust LS, and Residual Exclusion under NLOS Bias")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_nlos_weighting_demo(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "nlos_weighting_demo.png"
    data = robust_solver_results()
    true_positions = data["true_positions"]
    solutions = data["solutions"]
    nlos_counts = np.sum(data["nlos_mask"], axis=1)
    selected_epoch = int(np.argmax(nlos_counts))

    labels = ["OLS", "WLS", "Huber", "Cauchy", "Exclusion"]
    keys = ["ols", "wls", "huber", "cauchy", "residual_exclusion"]
    errors = [float(np.linalg.norm(solutions[key][selected_epoch] - true_positions[selected_epoch])) for key in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, errors, color=["#e76f51", "#2a9d8f", "#264653", "#0ea5e9", "#8a5cf6"])
    ax.set_ylabel("Positioning error at selected NLOS epoch (m)")
    ax.set_title(f"NLOS Weighting Demo (epoch {selected_epoch})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gnss_ins_comparison(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "gnss_ins_comparison.png"
    data = gnss_ins_results()
    true_positions = data["true_positions"]

    series = {
        "GNSS-only degraded": data["gnss_positions"],
        "INS/odometry-only": data["ins_only"],
        "GNSS/INS toy KF": data["kf_positions"],
    }
    colors = {"GNSS-only degraded": "#e76f51", "INS/odometry-only": "#f4a261", "GNSS/INS toy KF": "#2a9d8f"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, positions in series.items():
        errors = np.linalg.norm(positions - true_positions, axis=1)
        ax.plot(errors, label=label, color=colors[label], linewidth=1.8)
    ax.axvspan(32, 47, color="#e9c46a", alpha=0.25, label="GNSS degradation interval")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Positioning error (m)")
    ax.set_title("GNSS-only vs INS-only vs GNSS/INS Toy Kalman Filter")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _copy_plot(source: Path, target: Path) -> Path:
    if source.resolve() != target.resolve():
        copyfile(source, target)
    return target


def plot_solver_comparison_controlled(results_dir: Path | str | None = None) -> Path:
    output_dir = _results_dir(results_dir)
    return _copy_plot(plot_robust_ls_comparison(output_dir), output_dir / "solver_comparison_controlled.png")


def plot_nlos_uncertainty_weighting(results_dir: Path | str | None = None) -> Path:
    output_dir = _results_dir(results_dir)
    return _copy_plot(plot_nlos_weighting_demo(output_dir), output_dir / "nlos_uncertainty_weighting.png")


def plot_gnss_ins_kf_demo(results_dir: Path | str | None = None) -> Path:
    output_dir = _results_dir(results_dir)
    return _copy_plot(plot_gnss_ins_comparison(output_dir), output_dir / "gnss_ins_kf_demo.png")


def plot_leo_clean_geometry_comparison(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "leo_clean_geometry_comparison.png"
    data = clean_leo_comparison_results()

    fig, (ax_hdop, ax_error) = plt.subplots(1, 2, figsize=(12, 5))
    ax_hdop.plot(data["gnss_only"]["hdop"], label="Clean GNSS-only", color="#e76f51", linewidth=1.8)
    ax_hdop.plot(data["gnss_leo"]["hdop"], label="Clean GNSS + LEO-like", color="#2a9d8f", linewidth=1.8)
    ax_hdop.set_title("Clean Geometry: HDOP")
    ax_hdop.set_xlabel("Epoch")
    ax_hdop.set_ylabel("HDOP")
    ax_hdop.grid(alpha=0.25)
    ax_hdop.legend()

    ax_error.plot(data["gnss_only"]["errors"], label="Clean GNSS-only", color="#e76f51", linewidth=1.8)
    ax_error.plot(data["gnss_leo"]["errors"], label="Clean GNSS + LEO-like", color="#2a9d8f", linewidth=1.8)
    ax_error.set_title("Clean Geometry: Positioning Error")
    ax_error.set_xlabel("Epoch")
    ax_error.set_ylabel("Error (m)")
    ax_error.grid(alpha=0.25)
    ax_error.legend()

    fig.suptitle("LEO-like Aiding Under Clean Measurements")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_leo_degraded_aiding_comparison(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "leo_degraded_aiding_comparison.png"
    gnss = run_scenario("Multipath/NLOS", seed=18, measurement_noise_std=8.0, multipath_bias_level=80.0)
    aided = run_scenario(
        "GNSS + LEO-like aiding",
        enable_leo=True,
        num_leo=3,
        seed=18,
        measurement_noise_std=8.0,
        multipath_bias_level=80.0,
    )

    fig, (ax_error, ax_hdop) = plt.subplots(1, 2, figsize=(12, 5))
    ax_error.plot(gnss["position_errors"], label="Degraded GNSS-only", color="#e76f51", linewidth=1.8)
    ax_error.plot(aided["position_errors"], label="Degraded GNSS + LEO-like", color="#2a9d8f", linewidth=1.8)
    ax_error.set_title("Degraded Measurements: Error")
    ax_error.set_xlabel("Epoch")
    ax_error.set_ylabel("Error (m)")
    ax_error.grid(alpha=0.25)
    ax_error.legend()

    ax_hdop.plot(gnss["hdop"], label="GNSS-only HDOP", color="#e76f51", linewidth=1.8)
    ax_hdop.plot(aided["hdop"], label="GNSS + LEO-like HDOP", color="#2a9d8f", linewidth=1.8)
    ax_hdop.set_title("Degraded Measurements: HDOP")
    ax_hdop.set_xlabel("Epoch")
    ax_hdop.set_ylabel("HDOP")
    ax_hdop.grid(alpha=0.25)
    ax_hdop.legend()

    fig.suptitle("LEO-like Aiding Under Degraded GNSS Measurements")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _draw_urban_blocks(ax, blocks: list[UrbanBlock]) -> None:
    for block in blocks:
        width = block.max_x - block.min_x
        height = block.max_y - block.min_y
        rectangle = plt.Rectangle(
            (block.min_x, block.min_y),
            width,
            height,
            facecolor="#64748b",
            edgecolor="#334155",
            alpha=0.35,
        )
        ax.add_patch(rectangle)


def plot_urban_los_nlos_demo(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "urban_los_nlos_demo.png"
    data = urban_los_nlos_demo_results(seed=22)
    trajectory = data["trajectory"]
    nlos_count = data["nlos_count"]

    fig, (ax_map, ax_series) = plt.subplots(1, 2, figsize=(12, 5))
    _draw_urban_blocks(ax_map, data["blocks"])
    scatter = ax_map.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        c=nlos_count,
        cmap="magma",
        s=34,
        edgecolor="white",
        linewidth=0.3,
    )
    ax_map.plot(trajectory[:, 0], trajectory[:, 1], color="#0f172a", linewidth=0.9, alpha=0.7)
    ax_map.set_title("Synthetic Urban LOS/NLOS Labels")
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax_map, label="NLOS measurement count")

    ax_series.plot(nlos_count, color="#be123c", linewidth=1.8)
    ax_series.set_title("NLOS Count Along Receiver Trajectory")
    ax_series.set_xlabel("Epoch")
    ax_series.set_ylabel("NLOS count")
    ax_series.grid(alpha=0.25)

    fig.suptitle("Urban LOS/NLOS Toy Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_urban_error_map(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "urban_error_map.png"
    data = urban_error_map_results(grid_size=21, seed=22)
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    extent = [float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, grid, title in [
        (axes[0], data["ols_error_grid"], "OLS urban-grid error"),
        (axes[1], data["wls_error_grid"], "WLS with LOS/NLOS uncertainty"),
    ]:
        image = axis.imshow(grid, origin="lower", extent=extent, cmap="viridis", aspect="auto")
        _draw_urban_blocks(axis, data["blocks"])
        axis.set_title(title)
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        fig.colorbar(image, ax=axis, label="Positioning error (m)")

    fig.suptitle("Urban Error Map: Geometry-Based NLOS Toy Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_urban_hdop_map(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "urban_hdop_map.png"
    data = urban_error_map_results(grid_size=21, seed=22)
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    extent = [float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])]

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(data["hdop_grid"], origin="lower", extent=extent, cmap="cividis", aspect="auto")
    _draw_urban_blocks(ax, data["blocks"])
    ax.set_title("Urban Grid HDOP")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    fig.colorbar(image, ax=ax, label="HDOP")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sliding_window_demo(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "sliding_window_demo.png"
    data = sliding_window_results(seed=52)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["single_epoch_errors"], label="Single-epoch OLS", color="#e76f51", linewidth=1.8)
    ax.plot(data["sliding_errors"], label="Sliding-window LS", color="#2a9d8f", linewidth=1.8)
    ax.axvspan(
        data["degraded_start"],
        data["degraded_stop"],
        color="#e9c46a",
        alpha=0.25,
        label="Degraded interval",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Positioning error (m)")
    ax.set_title("Sliding-Window LS Toy Optimization")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_collaborative_positioning_demo(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "collaborative_positioning_demo.png"
    data = collaborative_results(seed=61)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["true_positions_a"][:, 0], data["true_positions_a"][:, 1], color="#0f172a", label="Agent A truth")
    ax.plot(data["true_positions_b"][:, 0], data["true_positions_b"][:, 1], color="#475569", label="Agent B truth")
    ax.plot(
        data["gnss_positions_b"][:, 0],
        data["gnss_positions_b"][:, 1],
        color="#e76f51",
        linestyle="--",
        label="Agent B GNSS-only",
    )
    ax.plot(
        data["collaborative_positions_b"][:, 0],
        data["collaborative_positions_b"][:, 1],
        color="#2a9d8f",
        linewidth=1.8,
        label="Agent B collaborative",
    )
    ax.set_title("Two-Agent Collaborative Positioning Toy Demo")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_collaborative_positioning_error(results_dir: Path | str | None = None) -> Path:
    path = _results_dir(results_dir) / "collaborative_positioning_error.png"
    data = collaborative_results(seed=61)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["gnss_errors_b"], label="Agent B GNSS-only", color="#e76f51", linewidth=1.8)
    ax.plot(data["collaborative_errors_b"], label="Agent B collaborative", color="#2a9d8f", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Positioning error (m)")
    ax.set_title("Collaborative Constraint Error Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_all_plots(results_dir: Path | str | None = None) -> list[Path]:
    """Generate all fixed experiment figures."""

    return [
        plot_scenario_comparison(results_dir),
        plot_dop_vs_error(results_dir),
        plot_clean_leo_hdop_comparison(results_dir),
        plot_clean_leo_error_comparison(results_dir),
        plot_robust_ls_comparison(results_dir),
        plot_nlos_weighting_demo(results_dir),
        plot_gnss_ins_comparison(results_dir),
        plot_solver_comparison_controlled(results_dir),
        plot_nlos_uncertainty_weighting(results_dir),
        plot_gnss_ins_kf_demo(results_dir),
        plot_leo_clean_geometry_comparison(results_dir),
        plot_leo_degraded_aiding_comparison(results_dir),
        plot_urban_los_nlos_demo(results_dir),
        plot_urban_error_map(results_dir),
        plot_urban_hdop_map(results_dir),
        plot_sliding_window_demo(results_dir),
        plot_collaborative_positioning_demo(results_dir),
        plot_collaborative_positioning_error(results_dir),
    ]


def main() -> None:
    for path in generate_all_plots():
        print(path)


if __name__ == "__main__":
    main()
