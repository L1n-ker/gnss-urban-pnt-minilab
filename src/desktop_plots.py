"""Matplotlib figures for the Chinese desktop interface."""

from __future__ import annotations

import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure


def configure_chinese_fonts() -> None:
    """Prefer common Windows Chinese fonts and keep minus signs readable."""

    rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    rcParams["axes.unicode_minus"] = False


def _new_figure(title: str, xlabel: str, ylabel: str) -> tuple[Figure, object]:
    fig = Figure(figsize=(6.6, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig, ax


def trajectory_figure(result: dict) -> Figure:
    """Create a true-vs-estimated trajectory figure."""

    fig, ax = _new_figure("真实轨迹与估计轨迹", "东向位置 (m)", "北向位置 (m)")
    true_positions = result["true_positions"]
    raw_positions = result["raw_gnss_positions"]
    estimated_positions = result["estimated_positions"]
    ins_positions = result.get("ins_positions")

    ax.plot(true_positions[:, 0], true_positions[:, 1], label="真实轨迹", color="#111827", linewidth=2.4)
    ax.plot(raw_positions[:, 0], raw_positions[:, 1], label="最小二乘估计", color="#2563eb", linewidth=1.8)

    if ins_positions is not None:
        ax.plot(ins_positions[:, 0], ins_positions[:, 1], label="INS 预测", color="#d97706", linestyle="--")
        ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], label="融合估计", color="#059669", linewidth=2.2)

    ax.scatter(true_positions[0, 0], true_positions[0, 1], color="#16a34a", s=45, label="起点")
    ax.scatter(true_positions[-1, 0], true_positions[-1, 1], color="#dc2626", s=45, label="终点")
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def error_figure(result: dict) -> Figure:
    """Create a positioning error time-series figure."""

    fig, ax = _new_figure("定位误差随时间变化", "时间步", "误差 (m)")
    errors = result["position_errors"]
    steps = np.arange(len(errors))
    flags = result["anomaly_flags"]

    ax.plot(steps, errors, label="定位误差", color="#dc2626", linewidth=1.8)
    if np.any(flags):
        ax.scatter(steps[flags], errors[flags], label="残差异常", color="#7c3aed", marker="x", s=42)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def residual_figure(result: dict) -> Figure:
    """Create a residual RMS figure."""

    fig, ax = _new_figure("最小二乘残差 RMS", "时间步", "残差 RMS (m)")
    residuals = result["residual_rms"]
    steps = np.arange(len(residuals))
    threshold = result["residual_threshold"]

    ax.plot(steps, residuals, label="残差 RMS", color="#0891b2", linewidth=1.8)
    ax.axhline(threshold, label="检测阈值", color="#9333ea", linestyle="--", linewidth=1.5)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def geometry_figure(result: dict) -> Figure:
    """Create a satellite geometry overview figure."""

    fig, ax = _new_figure("卫星几何示意", "东向位置 (km)", "北向位置 (km)")
    gnss_satellites = result["gnss_satellites"] / 1000.0
    true_positions = result["true_positions"] / 1000.0
    leo_positions = result.get("leo_positions")

    ax.scatter(gnss_satellites[:, 0], gnss_satellites[:, 1], label="GNSS 卫星", color="#2563eb", s=46)
    for index, sat_pos in enumerate(gnss_satellites, start=1):
        ax.text(sat_pos[0], sat_pos[1], f"G{index}", fontsize=9)

    if leo_positions is not None:
        for sat in range(leo_positions.shape[1]):
            path = leo_positions[:, sat, :] / 1000.0
            ax.plot(path[:, 0], path[:, 1], label=f"LEO 类卫星 {sat + 1}", linewidth=1.6)

    ax.plot(true_positions[:, 0], true_positions[:, 1], label="接收机轨迹", color="#111827", linewidth=2.0)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
