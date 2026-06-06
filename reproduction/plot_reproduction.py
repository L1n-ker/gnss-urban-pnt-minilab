"""Plot outputs for the simplified pseudorange-correlogram reproduction."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_reproduction_plots(results_dir: Path | str | None = None) -> list[Path]:
    """Create all reproduction figures from the saved NPZ arrays."""

    output_dir = Path(results_dir) if results_dir is not None else PROJECT_ROOT / "results" / "reproduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(output_dir / "correlogram_reproduction_arrays.npz")

    figure_paths = [
        _plot_error_timeseries(arrays, output_dir),
        _plot_trajectory(arrays, output_dir),
        _plot_score_map(arrays, output_dir, "reproduced_figure_score_map.png"),
        _plot_score_map(arrays, output_dir, "correlogram_xy_slice.png"),
        _plot_clean_vs_nlos(arrays, output_dir),
    ]
    return figure_paths


def _plot_error_timeseries(arrays: np.lib.npyio.NpzFile, output_dir: Path) -> Path:
    path = output_dir / "reproduced_figure_error_timeseries.png"
    epochs = np.arange(len(arrays["ols_errors"]))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, arrays["ols_errors"], label="OLS baseline", color="#e76f51", linewidth=1.8)
    ax1.plot(
        epochs,
        arrays["correlogram_errors"],
        label="Simplified pseudorange correlogram",
        color="#2a9d8f",
        linewidth=1.8,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Horizontal error (m)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.fill_between(epochs, arrays["nlos_counts"], color="#8a5cf6", alpha=0.16, label="NLOS-like count")
    ax2.set_ylabel("Synthetic NLOS-like measurements")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.suptitle("Simplified Reproduction: OLS vs Pseudorange Correlogram")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_trajectory(arrays: np.lib.npyio.NpzFile, output_dir: Path) -> Path:
    path = output_dir / "reproduced_figure_trajectory.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(arrays["true_positions"][:, 0], arrays["true_positions"][:, 1], label="UrbanNav public GT route", color="#264653", linewidth=2.0)
    ax.plot(arrays["ols_positions"][:, 0], arrays["ols_positions"][:, 1], label="OLS baseline", color="#e76f51", alpha=0.8)
    ax.plot(
        arrays["correlogram_positions"][:, 0],
        arrays["correlogram_positions"][:, 1],
        label="Simplified correlogram",
        color="#2a9d8f",
        alpha=0.85,
    )
    ax.set_xlabel("Local east (m)")
    ax.set_ylabel("Local north (m)")
    ax.set_title("UrbanNav Route Subset with Synthetic Position Estimates")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_score_map(arrays: np.lib.npyio.NpzFile, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    selected_epoch = int(arrays["selected_epoch"])
    score_grid = arrays["score_grid"]
    x_grid = arrays["x_grid"]
    y_grid = arrays["y_grid"]
    best_clock_bias_m = float(arrays["best_clock_bias_m"])

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        score_grid,
        origin="lower",
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        aspect="auto",
        cmap="viridis",
    )
    ax.scatter(
        arrays["true_positions"][selected_epoch, 0],
        arrays["true_positions"][selected_epoch, 1],
        marker="x",
        s=80,
        color="white",
        label="True route point",
    )
    ax.scatter(
        arrays["ols_positions"][selected_epoch, 0],
        arrays["ols_positions"][selected_epoch, 1],
        marker="o",
        s=50,
        color="#e76f51",
        label="OLS estimate",
    )
    ax.scatter(
        arrays["correlogram_positions"][selected_epoch, 0],
        arrays["correlogram_positions"][selected_epoch, 1],
        marker="^",
        s=60,
        color="#2a9d8f",
        label="Correlogram estimate",
    )
    ax.set_xlabel("Local east (m)")
    ax.set_ylabel("Local north (m)")
    ax.set_title(f"XY Score Slice at Epoch {selected_epoch}, Clock Bias {best_clock_bias_m:.1f} m")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Weighted correlogram score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_clean_vs_nlos(arrays: np.lib.npyio.NpzFile, output_dir: Path) -> Path:
    path = output_dir / "reproduction_clean_vs_nlos.png"
    labels = ["Clean/open-sky-like", "NLOS-dominated"]
    ols_means = [
        float(np.mean(arrays["clean_ols_errors"])),
        float(np.mean(arrays["nlos_ols_errors"])),
    ]
    corr_means = [
        float(np.mean(arrays["clean_correlogram_errors"])),
        float(np.mean(arrays["nlos_correlogram_errors"])),
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.18, ols_means, width=0.36, label="OLS baseline", color="#e76f51")
    ax.bar(
        x + 0.18,
        corr_means,
        width=0.36,
        label="Pseudorange-correlogram toy",
        color="#2a9d8f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean horizontal error (m)")
    ax.set_title("Clean vs NLOS Synthetic-Measurement Scenarios")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    for path in generate_reproduction_plots():
        print(path)


if __name__ == "__main__":
    main()
