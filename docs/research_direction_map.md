# Research Direction Map

Project identity: **GNSS Urban PNT MiniLab: Robust Positioning, Urban Degradation, and Assisted Navigation**.

This document maps the implemented modules to research-preparation themes. The mapping is intentionally careful: the project is a simplified educational simulator and does not claim real receiver, SDR, or full paper-reproduction capability.

## Direction A - Robust GNSS Under Urban Degradation

Implemented:

- `src/gnss_solver.py`: OLS, WLS, Huber IRLS, and residual-exclusion positioning baselines.
- `src/urban_environment.py`: 2D rectangular-building LOS/NLOS toy model, NLOS measurement sigmas, and urban error-map generation.
- `experiments/run_all_experiments.py`: controlled solver, urban LOS/NLOS, and urban error-map outputs.

Generated outputs:

- `results/solver_comparison_controlled.png`
- `results/solver_comparison_metrics.csv`
- `results/nlos_uncertainty_weighting.png`
- `results/urban_los_nlos_demo.png`
- `results/urban_los_nlos_labels.csv`
- `results/urban_error_map.png`
- `results/urban_hdop_map.png`
- `results/urban_error_map.csv`

Simplified:

- Buildings are 2D rectangles, not 3D city models.
- LOS/NLOS labels are geometry-derived toy labels, not learned or surveyed labels.
- WLS benefits from known uncertainty labels; this is an educational assumption.

## Direction B - Assisted PNT And GNSS/INS Toy Fusion

Implemented:

- `src/ins_filter.py`: synthetic velocity generation, dead reckoning, and constant-velocity Kalman filtering.
- `experiments/experiment_data.py`: deterministic GNSS degradation interval for GNSS-only, odometry-only, and GNSS/INS toy-KF comparison.

Generated outputs:

- `results/gnss_ins_kf_demo.png`
- `results/gnss_ins_kf_metrics.csv`

Simplified:

- Synthetic odometry is generated from simulated truth with noise for learning.
- The filter is not strapdown inertial mechanization or an error-state EKF.

## Direction C - LEO-Like And Multi-Source PNT Extensions

Implemented:

- `src/leo_aiding.py`: abstract moving LEO-like ranging sources.
- `src/dop.py`: HDOP/GDOP-style simplified geometry metrics.
- Clean and degraded LEO-like comparisons.

Generated outputs:

- `results/leo_clean_geometry_comparison.png`
- `results/leo_degraded_aiding_comparison.png`
- `results/leo_aiding_metrics.csv`

Simplified:

- LEO-like sources are conceptual range beacons, not real orbit propagation, ephemerides, Doppler, or signal models.
- Lower HDOP does not guarantee lower error under biased measurements.

## Direction D - Sliding-Window / Factor-Graph-Inspired Optimization

Implemented:

- `src/sliding_window.py`: a lightweight sliding-window least-squares optimizer with pseudorange residuals, odometry smoothness constraints, and a first-state prior.

Generated outputs:

- `results/sliding_window_demo.png`
- `results/sliding_window_metrics.csv`

Simplified:

- This is factor-graph-inspired but not a complete factor graph library or GTSAM-style implementation.
- The odometry constraint is synthetic and local.

## Direction E - Collaborative / Cooperative Positioning

Implemented:

- `src/collaborative_positioning.py`: two-agent joint least squares with each agent's GNSS pseudoranges and one inter-agent distance constraint.

Generated outputs:

- `results/collaborative_positioning_demo.png`
- `results/collaborative_positioning_error.png`
- `results/collaborative_positioning_metrics.csv`

Simplified:

- This is a two-agent conceptual example, not a full V2X, UWB, or networked cooperative localization system.

## Direction F - Literature-Grounded Conceptual Replication

Implemented:

- `src/pseudorange_correlogram.py`
- `reproduction/run_reproduction.py`
- `reproduction/selected_paper_summary.md`

Generated outputs:

- `results/reproduction/reproduction_metrics.csv`
- `results/reproduction/reproduced_figure_error_timeseries.png`
- `results/reproduction/reproduced_figure_trajectory.png`
- `results/reproduction/reproduced_figure_score_map.png`

Simplified:

- The public UrbanNav route is used as a reference trajectory.
- Pseudoranges, C/N0, clock bias, and NLOS-like biases are synthetic.
- This is a conceptual replication of one algorithmic idea, not an exact reproduction of the selected paper's real-data pipeline.
