"""Chinese desktop UI labels and formatting helpers."""

from __future__ import annotations

from collections import OrderedDict


SCENARIO_LABELS = OrderedDict(
    [
        ("理想 GNSS", "Ideal GNSS"),
        ("多路径/NLOS", "Multipath/NLOS"),
        ("类欺骗漂移", "Spoofing-like drift"),
        ("GNSS + INS 辅助", "GNSS + INS"),
        ("GNSS + LEO 类辅助", "GNSS + LEO-like aiding"),
        ("GNSS + INS + LEO 类辅助", "GNSS + INS + LEO-like aiding"),
    ]
)

CHINESE_SCENARIOS = list(SCENARIO_LABELS.keys())
INTERNAL_TO_CHINESE = {internal: chinese for chinese, internal in SCENARIO_LABELS.items()}

METRIC_LABELS = OrderedDict(
    [
        ("mean_error_m", "平均定位误差 (m)"),
        ("median_error_m", "中位定位误差 (m)"),
        ("rmse_error_m", "定位 RMSE (m)"),
        ("max_error_m", "最大定位误差 (m)"),
        ("final_error_m", "末端定位误差 (m)"),
        ("mean_residual_rms_m", "平均残差 RMS (m)"),
        ("max_residual_rms_m", "最大残差 RMS (m)"),
        ("mean_hdop", "平均 HDOP"),
        ("max_hdop", "最大 HDOP"),
        ("anomaly_count", "异常历元数"),
        ("convergence_rate_pct", "收敛率 (%)"),
    ]
)


def scenario_to_internal(chinese_label: str) -> str:
    """Map a Chinese UI scenario label to the simulation scenario name."""

    return SCENARIO_LABELS.get(chinese_label, chinese_label)


def scenario_to_chinese(internal_label: str) -> str:
    """Map an internal simulation scenario name back to a Chinese label."""

    return INTERNAL_TO_CHINESE.get(internal_label, internal_label)


def build_metric_rows(metrics: dict[str, float]) -> list[tuple[str, str]]:
    """Format metric dictionary values for a Tkinter table."""

    rows: list[tuple[str, str]] = []
    for key, label in METRIC_LABELS.items():
        if key not in metrics:
            continue
        value = metrics[key]
        rows.append((label, f"{float(value):.2f}"))
    return rows
