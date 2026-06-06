"""Chinese desktop interface for GNSS Robustness MiniLab.

Run with:

    python app.py

The program uses Tkinter and Matplotlib, so it opens a native desktop window
instead of requiring a web browser.
"""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.desktop_config import (
    CHINESE_SCENARIOS,
    build_metric_rows,
    scenario_to_chinese,
    scenario_to_internal,
)
from src.desktop_plots import (
    configure_chinese_fonts,
    error_figure,
    geometry_figure,
    residual_figure,
    trajectory_figure,
)
from src.result_artifacts import (
    get_result_path,
    load_summary_metrics,
    missing_results_message,
    project_root_from_here,
    result_file_exists,
)
from src.simulation import run_comparison, run_scenario


class DesktopMiniLab(tk.Tk):
    """Native Tkinter application for the MiniLab."""

    def __init__(self) -> None:
        super().__init__()
        configure_chinese_fonts()
        self.title("GNSS 鲁棒性迷你实验室")
        self.geometry("1280x820")
        self.minsize(1120, 720)

        self.result: dict | None = None
        self.canvases: dict[str, FigureCanvasTkAgg] = {}
        self.summary_tree: ttk.Treeview | None = None
        self.project_root = project_root_from_here()

        self._build_variables()
        self._build_styles()
        self._build_layout()
        self._sync_aiding_from_scenario()
        self.run_current_scenario()

    def _build_variables(self) -> None:
        self.scenario_var = tk.StringVar(value=CHINESE_SCENARIOS[0])
        self.noise_var = tk.DoubleVar(value=5.0)
        self.multipath_var = tk.DoubleVar(value=45.0)
        self.nlos_var = tk.DoubleVar(value=0.25)
        self.spoof_start_var = tk.IntVar(value=45)
        self.spoof_drift_var = tk.DoubleVar(value=2.0)
        self.enable_ins_var = tk.BooleanVar(value=False)
        self.enable_leo_var = tk.BooleanVar(value=False)
        self.num_leo_var = tk.IntVar(value=2)
        self.seed_var = tk.IntVar(value=4)
        self.scenario_var.trace_add("write", lambda *_: self._sync_aiding_from_scenario())

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f8fafc")
        style.configure("Panel.TFrame", background="#eef2f7")
        style.configure("TLabel", background="#f8fafc", foreground="#111827", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Microsoft YaHei UI", 10), foreground="#475569")
        style.configure("Panel.TLabel", background="#eef2f7")
        style.configure("TButton", font=("Microsoft YaHei UI", 10))
        style.configure("Treeview", rowheight=28, font=("Microsoft YaHei UI", 10))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=14)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(header, text="GNSS 鲁棒性迷你实验室", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="二维伪距定位、测量级异常、INS 预测与 LEO 类辅助演示。仅用于教育仿真，不生成或发射真实 GNSS 信号。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        controls = ttk.Frame(container, style="Panel.TFrame", padding=12)
        controls.grid(row=1, column=0, sticky="nsw", padx=(0, 12))
        self._build_controls(controls)

        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=1, column=1, sticky="nsew")
        self._build_tabs()

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="实验控制", style="Panel.TLabel", font=("Microsoft YaHei UI", 13, "bold")).pack(anchor="w")

        self._add_combo(parent, "场景选择", self.scenario_var, CHINESE_SCENARIOS)
        self._add_scale(parent, "测量噪声 (m)", self.noise_var, 0.0, 30.0, 0.5)
        self._add_scale(parent, "多路径偏差 (m)", self.multipath_var, 0.0, 180.0, 5.0)
        self._add_scale(parent, "NLOS 概率", self.nlos_var, 0.0, 1.0, 0.05)
        self._add_scale(parent, "类欺骗起始时刻", self.spoof_start_var, 0, 79, 1)
        self._add_scale(parent, "类欺骗漂移强度 (m/步)", self.spoof_drift_var, 0.0, 10.0, 0.25)

        ttk.Checkbutton(parent, text="启用 INS 辅助", variable=self.enable_ins_var).pack(anchor="w", pady=(8, 2))
        ttk.Checkbutton(parent, text="启用 LEO 类辅助", variable=self.enable_leo_var).pack(anchor="w", pady=2)

        leo_frame = ttk.Frame(parent, style="Panel.TFrame")
        leo_frame.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(leo_frame, text="LEO 类卫星数量", style="Panel.TLabel").pack(anchor="w")
        ttk.Spinbox(leo_frame, from_=0, to=5, textvariable=self.num_leo_var, width=8).pack(anchor="w", pady=(3, 0))

        seed_frame = ttk.Frame(parent, style="Panel.TFrame")
        seed_frame.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(seed_frame, text="随机种子", style="Panel.TLabel").pack(anchor="w")
        ttk.Spinbox(seed_frame, from_=0, to=9999, textvariable=self.seed_var, width=8).pack(anchor="w", pady=(3, 0))

        ttk.Button(parent, text="运行仿真", command=self.run_current_scenario).pack(fill=tk.X, pady=(14, 6))
        ttk.Button(parent, text="恢复默认参数", command=self.reset_defaults).pack(fill=tk.X)

        note = (
            "安全说明：本程序只修改 Python 数组中的合成伪距，"
            "不包含 RF 信号生成、SDR 发射或真实欺骗能力。"
        )
        ttk.Label(parent, text=note, style="Panel.TLabel", wraplength=250, foreground="#475569").pack(anchor="w", pady=(18, 0))

    def _add_combo(self, parent: ttk.Frame, label: str, variable: tk.StringVar, values: list[str]) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(frame, text=label, style="Panel.TLabel").pack(anchor="w")
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=28)
        combo.pack(fill=tk.X, pady=(3, 0))

    def _add_scale(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        from_value: float,
        to_value: float,
        resolution: float,
    ) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill=tk.X, pady=(8, 0))
        value_label = ttk.Label(frame, style="Panel.TLabel")

        def update_value(*_: object) -> None:
            value = variable.get()
            if isinstance(value, float):
                value_label.configure(text=f"{label}: {value:.2f}")
            else:
                value_label.configure(text=f"{label}: {value}")

        update_value()
        value_label.pack(anchor="w")
        scale = tk.Scale(
            frame,
            variable=variable,
            from_=from_value,
            to=to_value,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=245,
            bg="#eef2f7",
            highlightthickness=0,
            command=lambda _: update_value(),
        )
        scale.pack(fill=tk.X)

    def _build_tabs(self) -> None:
        self.figure_tabs: dict[str, ttk.Frame] = {}
        for key, label in [
            ("trajectory", "轨迹对比"),
            ("error", "误差曲线"),
            ("residual", "残差检测"),
            ("geometry", "卫星几何"),
        ]:
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text=label)
            self.figure_tabs[key] = frame

        metrics_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(metrics_frame, text="指标表")
        self._build_metric_tables(metrics_frame)

        self._build_research_result_tabs()

    def _build_metric_tables(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

        ttk.Label(parent, text="当前场景指标", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.metric_tree = ttk.Treeview(parent, columns=("metric", "value"), show="headings", height=9)
        self.metric_tree.heading("metric", text="指标")
        self.metric_tree.heading("value", text="数值")
        self.metric_tree.column("metric", width=240, anchor="w")
        self.metric_tree.column("value", width=120, anchor="center")
        self.metric_tree.grid(row=1, column=0, sticky="nsew", pady=(6, 14))

        ttk.Label(parent, text="多场景对比", font=("Microsoft YaHei UI", 12, "bold")).grid(row=2, column=0, sticky="w")
        columns = ("scenario", "mean", "median", "max", "residual", "anomaly")
        self.comparison_tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        headings = {
            "scenario": "场景",
            "mean": "平均误差 (m)",
            "median": "中位误差 (m)",
            "max": "最大误差 (m)",
            "residual": "平均残差 RMS (m)",
            "anomaly": "异常历元数",
        }
        widths = {"scenario": 220, "mean": 115, "median": 115, "max": 115, "residual": 145, "anomaly": 100}
        for column in columns:
            self.comparison_tree.heading(column, text=headings[column])
            self.comparison_tree.column(column, width=widths[column], anchor="center")
        self.comparison_tree.column("scenario", anchor="w")
        self.comparison_tree.grid(row=3, column=0, sticky="nsew", pady=(6, 0))

    def _build_research_result_tabs(self) -> None:
        self._build_image_result_tab(
            tab_title="DOP / HDOP Analysis",
            canvas_key="research_dop",
            explanation=(
                "HDOP reflects satellite/source geometry, not total positioning error. "
                "LEO-like aiding can reduce HDOP by adding observation geometry, but "
                "instantaneous error also depends on noise, multipath/NLOS bias, anomaly drift, "
                "and solver robustness."
            ),
            figures=[
                ("HDOP vs Error", "dop_vs_error.png"),
                ("Clean LEO HDOP", "clean_leo_hdop_comparison.png"),
                ("Clean LEO Error", "clean_leo_error_comparison.png"),
            ],
        )
        self._build_image_result_tab(
            tab_title="LEO-like Aiding",
            canvas_key="research_leo",
            explanation=(
                "This tab separates clean geometry improvement from degraded-measurement behavior. "
                "LEO-like sources can reduce HDOP in this conceptual model, but positioning error "
                "can still remain high when biased measurements are present. This is not a real LEO PNT system."
            ),
            figures=[
                ("Clean LEO Geometry", "leo_clean_geometry_comparison.png"),
                ("Degraded LEO Aiding", "leo_degraded_aiding_comparison.png"),
            ],
        )
        self._build_image_result_tab(
            tab_title="Robust LS Comparison",
            canvas_key="research_robust",
            explanation=(
                "OLS is the baseline. WLS can perform well when measurement uncertainty or "
                "NLOS labels are available. Huber IRLS and residual exclusion are simplified "
                "educational baselines; their behavior depends on threshold/scale tuning and "
                "initial solution quality."
            ),
            figures=[
                ("Controlled Solver Comparison", "solver_comparison_controlled.png"),
                ("NLOS Uncertainty Weighting", "nlos_uncertainty_weighting.png"),
            ],
        )
        self._build_image_result_tab(
            tab_title="Urban LOS/NLOS Demo",
            canvas_key="research_urban_los",
            explanation=(
                "This tab shows a simplified 2D urban-canyon model. Rectangular buildings "
                "block receiver-to-source line segments, creating synthetic LOS/NLOS labels. "
                "NLOS measurements receive larger positive pseudorange bias and larger "
                "uncertainty for WLS weighting. This is not a full 3D mapping-aided GNSS model."
            ),
            figures=[("Urban LOS/NLOS", "urban_los_nlos_demo.png")],
        )
        self._build_image_result_tab(
            tab_title="GNSS/INS Toy Fusion",
            canvas_key="research_ins",
            explanation=(
                "The INS/odometry data here are synthetic velocity measurements generated "
                "from the simulated ground-truth trajectory with noise. The Kalman filter is "
                "a toy constant-velocity model for demonstrating short-term aiding during "
                "GNSS degradation, not a complete strapdown INS or full GNSS/INS EKF."
            ),
            figures=[("GNSS/INS KF Demo", "gnss_ins_kf_demo.png")],
        )
        self._build_image_result_tab(
            tab_title="Urban Error Map",
            canvas_key="research_urban_map",
            explanation=(
                "The urban error map evaluates OLS and WLS at each point of a synthetic 2D grid. "
                "It demonstrates how geometry-based NLOS labels and uncertainty weighting can "
                "change positioning error across an urban-like area. The heatmaps are conceptual "
                "simulation outputs, not real city-map validation."
            ),
            figures=[
                ("Urban Error Map", "urban_error_map.png"),
                ("Urban HDOP Map", "urban_hdop_map.png"),
            ],
        )
        self._build_image_result_tab(
            tab_title="Sliding-window Optimization",
            canvas_key="research_sliding_window",
            explanation=(
                "This tab shows a simplified sliding-window least-squares optimizer inspired by "
                "factor-graph GNSS/INS research. It combines pseudorange residuals with short-term "
                "odometry smoothness constraints. It is not a full factor graph implementation."
            ),
            figures=[("Sliding-window LS", "sliding_window_demo.png")],
        )
        self._build_image_result_tab(
            tab_title="Collaborative Positioning",
            canvas_key="research_collaborative",
            explanation=(
                "This tab shows a two-agent toy example where a relative distance constraint helps "
                "stabilize a degraded GNSS solution. It is a conceptual cooperative-localization "
                "demo, not a full V2X or production collaborative positioning system."
            ),
            figures=[
                ("Collaborative Trajectory", "collaborative_positioning_demo.png"),
                ("Collaborative Error", "collaborative_positioning_error.png"),
            ],
        )
        self._build_image_result_tab(
            tab_title="Pseudorange Correlogram Reproduction",
            canvas_key="research_reproduction",
            explanation=(
                "This tab loads the simplified literature-grounded reproduction inspired by "
                "Vicenzo, Xu, Xu, and Hsu (2024). It uses a public UrbanNav route subset but "
                "synthetic pseudorange/C/N0 measurements, so it should be described as a "
                "conceptual reproduction of one algorithmic idea rather than a full paper reproduction."
            ),
            figures=[
                ("Error Time Series", "reproduction/reproduced_figure_error_timeseries.png"),
                ("Trajectory", "reproduction/reproduced_figure_trajectory.png"),
                ("Score Map", "reproduction/reproduced_figure_score_map.png"),
            ],
        )
        self._build_experiment_summary_tab()

    def _build_image_result_tab(
        self,
        tab_title: str,
        canvas_key: str,
        explanation: str,
        figures: list[tuple[str, str]],
    ) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=tab_title)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text=explanation, wraplength=980, foreground="#334155").grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        figure_notebook = ttk.Notebook(frame)
        figure_notebook.grid(row=1, column=0, sticky="nsew")
        for index, (label, filename) in enumerate(figures):
            image_frame = ttk.Frame(figure_notebook, padding=8)
            figure_notebook.add(image_frame, text=label)
            figure = self._result_image_figure(filename)
            key = f"{canvas_key}_{index}"
            canvas = FigureCanvasTkAgg(figure, master=image_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvases[key] = canvas

    def _build_experiment_summary_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Experiment Summary")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        explanation = (
            "This tab loads fixed experiment outputs from the results folder. "
            "Changing the left-side parameters does not update these pre-generated figures. "
            "Use the button to regenerate summary_metrics.csv and all PNG figures."
        )
        ttk.Label(frame, text=explanation, wraplength=980, foreground="#334155").grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )
        ttk.Button(frame, text="Run reproducible experiments", command=self.run_reproducible_experiments).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(0, 8),
        )

        body = ttk.Frame(frame)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=1)

        figure_frame = ttk.Frame(body, padding=(0, 0, 0, 8))
        figure_frame.grid(row=0, column=0, sticky="nsew")
        figure = self._result_image_figure("scenario_comparison.png")
        canvas = FigureCanvasTkAgg(figure, master=figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases["research_summary_scenario"] = canvas

        table_frame = ttk.Frame(body)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = (
            "family",
            "case",
            "mean",
            "median",
            "max",
            "rmse",
            "hdop",
        )
        self.summary_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        headings = {
            "family": "Experiment",
            "case": "Case",
            "mean": "Mean error",
            "median": "Median",
            "max": "Max",
            "rmse": "RMSE",
            "hdop": "Mean HDOP",
        }
        widths = {"family": 190, "case": 190, "mean": 90, "median": 90, "max": 90, "rmse": 90, "hdop": 90}
        for column in columns:
            self.summary_tree.heading(column, text=headings[column])
            self.summary_tree.column(column, width=widths[column], anchor="center")
        self.summary_tree.column("family", anchor="w")
        self.summary_tree.column("case", anchor="w")
        self.summary_tree.grid(row=0, column=0, sticky="nsew")
        self._load_summary_table()

    def _result_image_figure(self, filename: str) -> Figure:
        figure = Figure(figsize=(8.2, 4.8), dpi=100)
        axis = figure.add_subplot(111)
        axis.axis("off")
        path = get_result_path(filename, self.project_root)
        if not result_file_exists(filename, self.project_root):
            axis.text(
                0.5,
                0.5,
                f"{filename} is missing.\n{missing_results_message()}",
                ha="center",
                va="center",
                fontsize=12,
                color="#475569",
                transform=axis.transAxes,
            )
        else:
            image = mpimg.imread(path)
            axis.imshow(image)
            axis.set_title(filename)
        figure.tight_layout()
        return figure

    def _load_summary_table(self) -> None:
        if self.summary_tree is None:
            return
        self.summary_tree.delete(*self.summary_tree.get_children())
        try:
            summary = load_summary_metrics(self.project_root)
        except FileNotFoundError:
            self.summary_tree.insert("", tk.END, values=("Missing results", missing_results_message(), "", "", "", "", ""))
            return

        for _, row in summary.iterrows():
            mean_hdop = row.get("mean_hdop")
            hdop_text = "" if mean_hdop != mean_hdop else f"{float(mean_hdop):.2f}"
            self.summary_tree.insert(
                "",
                tk.END,
                values=(
                    row["experiment_family"],
                    row["case"],
                    f"{float(row['mean_positioning_error_m']):.2f}",
                    f"{float(row['median_positioning_error_m']):.2f}",
                    f"{float(row['max_positioning_error_m']):.2f}",
                    f"{float(row['rmse_positioning_error_m']):.2f}",
                    hdop_text,
                ),
            )

    def run_reproducible_experiments(self) -> None:
        script = self.project_root / "experiments" / "run_all_experiments.py"
        try:
            subprocess.run([sys.executable, str(script)], cwd=self.project_root, check=True)
        except Exception as exc:
            messagebox.showerror("实验生成失败", f"运行可复现实验时出现错误：\n{exc}")
            return
        messagebox.showinfo("实验生成完成", "已重新生成 results 文件夹中的 CSV 和 PNG 成果图。请重启程序或重新打开相应标签查看最新图片。")
        self._load_summary_table()

    def _sync_aiding_from_scenario(self) -> None:
        internal = scenario_to_internal(self.scenario_var.get())
        self.enable_ins_var.set("INS" in internal)
        self.enable_leo_var.set("LEO" in internal)

    def reset_defaults(self) -> None:
        self.scenario_var.set(CHINESE_SCENARIOS[0])
        self.noise_var.set(5.0)
        self.multipath_var.set(45.0)
        self.nlos_var.set(0.25)
        self.spoof_start_var.set(45)
        self.spoof_drift_var.set(2.0)
        self.num_leo_var.set(2)
        self.seed_var.set(4)
        self.run_current_scenario()

    def run_current_scenario(self) -> None:
        try:
            internal_scenario = scenario_to_internal(self.scenario_var.get())
            result = run_scenario(
                scenario=internal_scenario,
                measurement_noise_std=float(self.noise_var.get()),
                multipath_bias_level=float(self.multipath_var.get()),
                nlos_probability=float(self.nlos_var.get()),
                spoofing_start_time=int(self.spoof_start_var.get()),
                spoofing_drift_strength=float(self.spoof_drift_var.get()),
                enable_ins=bool(self.enable_ins_var.get()),
                enable_leo=bool(self.enable_leo_var.get()),
                num_leo=int(self.num_leo_var.get()),
                seed=int(self.seed_var.get()),
            )
            comparison = run_comparison(
                measurement_noise_std=float(self.noise_var.get()),
                multipath_bias_level=float(self.multipath_var.get()),
                nlos_probability=float(self.nlos_var.get()),
                spoofing_start_time=int(self.spoof_start_var.get()),
                spoofing_drift_strength=float(self.spoof_drift_var.get()),
                num_leo=int(self.num_leo_var.get()),
                seed=int(self.seed_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("运行失败", f"仿真运行时出现错误：\n{exc}")
            return

        self.result = result
        self._render_figures(result)
        self._render_metrics(result, comparison)

    def _render_figures(self, result: dict) -> None:
        figures = {
            "trajectory": trajectory_figure(result),
            "error": error_figure(result),
            "residual": residual_figure(result),
            "geometry": geometry_figure(result),
        }

        for key, figure in figures.items():
            frame = self.figure_tabs[key]
            old_canvas = self.canvases.get(key)
            if old_canvas is not None:
                old_canvas.get_tk_widget().destroy()

            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvases[key] = canvas

    def _render_metrics(self, result: dict, comparison: list[dict]) -> None:
        self.metric_tree.delete(*self.metric_tree.get_children())
        for metric, value in build_metric_rows(result["metrics"]):
            self.metric_tree.insert("", tk.END, values=(metric, value))

        self.comparison_tree.delete(*self.comparison_tree.get_children())
        for row in comparison:
            self.comparison_tree.insert(
                "",
                tk.END,
                values=(
                    scenario_to_chinese(str(row["scenario"])),
                    f"{float(row['mean_error_m']):.2f}",
                    f"{float(row['median_error_m']):.2f}",
                    f"{float(row['max_error_m']):.2f}",
                    f"{float(row['mean_residual_rms_m']):.2f}",
                    f"{float(row['anomaly_count']):.0f}",
                ),
            )


def main() -> None:
    app = DesktopMiniLab()
    app.mainloop()


if __name__ == "__main__":
    main()
