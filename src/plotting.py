"""Plotly figures used by the Streamlit app."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _simple_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_trajectories(result: dict) -> go.Figure:
    """Plot true, estimated, and optional aiding trajectories."""

    true_positions = result["true_positions"]
    estimated_positions = result["estimated_positions"]
    raw_positions = result["raw_gnss_positions"]
    ins_positions = result.get("ins_positions")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=true_positions[:, 0],
            y=true_positions[:, 1],
            mode="lines",
            name="True trajectory",
            line=dict(color="#111827", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=raw_positions[:, 0],
            y=raw_positions[:, 1],
            mode="lines",
            name="Least-squares estimate",
            line=dict(color="#2563eb", width=2),
        )
    )
    if ins_positions is not None:
        fig.add_trace(
            go.Scatter(
                x=ins_positions[:, 0],
                y=ins_positions[:, 1],
                mode="lines",
                name="INS prediction",
                line=dict(color="#f59e0b", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=estimated_positions[:, 0],
                y=estimated_positions[:, 1],
                mode="lines",
                name="Fused estimate",
                line=dict(color="#059669", width=3),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[true_positions[0, 0], true_positions[-1, 0]],
            y=[true_positions[0, 1], true_positions[-1, 1]],
            mode="markers",
            name="Start / end",
            marker=dict(color=["#22c55e", "#ef4444"], size=10),
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _simple_layout(fig, "True vs Estimated Receiver Trajectory", "East position (m)", "North position (m)")


def plot_error_time_series(result: dict) -> go.Figure:
    """Plot positioning error over time."""

    errors = result["position_errors"]
    steps = np.arange(len(errors))
    flags = result["anomaly_flags"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=errors,
            mode="lines",
            name="Position error",
            line=dict(color="#dc2626", width=2),
        )
    )
    if np.any(flags):
        fig.add_trace(
            go.Scatter(
                x=steps[flags],
                y=errors[flags],
                mode="markers",
                name="Residual anomaly",
                marker=dict(color="#7c3aed", size=8, symbol="x"),
            )
        )
    return _simple_layout(fig, "Positioning Error Over Time", "Time step", "Error (m)")


def plot_residuals(result: dict) -> go.Figure:
    """Plot residual RMS and detector threshold."""

    residuals = result["residual_rms"]
    steps = np.arange(len(residuals))
    threshold = result["residual_threshold"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=residuals,
            mode="lines",
            name="Residual RMS",
            line=dict(color="#0891b2", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=np.full_like(residuals, threshold),
            mode="lines",
            name="Detection threshold",
            line=dict(color="#9333ea", width=2, dash="dash"),
        )
    )
    return _simple_layout(fig, "Least-Squares Residual RMS", "Time step", "Residual RMS (m)")


def plot_satellite_geometry(result: dict) -> go.Figure:
    """Plot GNSS and optional LEO-like measurement source geometry."""

    gnss_satellites = result["gnss_satellites"] / 1000.0
    true_positions = result["true_positions"] / 1000.0
    leo_positions = result.get("leo_positions")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gnss_satellites[:, 0],
            y=gnss_satellites[:, 1],
            mode="markers+text",
            text=[f"G{i + 1}" for i in range(len(gnss_satellites))],
            textposition="top center",
            name="GNSS satellites",
            marker=dict(color="#2563eb", size=10),
        )
    )
    if leo_positions is not None:
        for sat in range(leo_positions.shape[1]):
            path = leo_positions[:, sat, :] / 1000.0
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode="lines",
                    name=f"LEO-like {sat + 1}",
                    line=dict(width=2),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=true_positions[:, 0],
            y=true_positions[:, 1],
            mode="lines",
            name="Receiver path near origin",
            line=dict(color="#111827", width=3),
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _simple_layout(fig, "Satellite Geometry Overview", "East position (km)", "North position (km)")
