# Relation to Papers and Research Directions I Read

This document maps the MiniLab to research directions I am preparing to study. It does not claim that the project reproduces any specific paper. The current implementation is a simplified educational simulation and should be understood as a conceptual learning bridge.

## Prof. Weisong Wen-Related Directions

Prof. Wen's research directions are closely connected to urban navigation, multi-sensor positioning, GNSS/INS integration, robust estimation, LiDAR/visual SLAM, and autonomous systems.

High-level ideas reflected in this project:

- **Urban navigation degradation:** the MiniLab models synthetic multipath/NLOS positive pseudorange bias.
- **Urban LOS/NLOS geometry:** the MiniLab now includes a 2D rectangular-building blockage toy model and urban error map.
- **GNSS/INS integration:** the project includes synthetic odometry generation, dead reckoning, and a toy GNSS/INS constant-velocity Kalman filter.
- **Robust estimation:** the project compares OLS, WLS, Huber IRLS, and residual-based exclusion under biased measurements.
- **Geometry-aware positioning:** the DOP/HDOP module helps connect source geometry with positioning stability.
- **Optimization and collaboration:** the project includes sliding-window least squares and a two-agent relative-distance constraint demo as conceptual extensions.
- **Future extension:** the current solver could later be extended toward factor-graph-inspired GNSS/INS optimization or public urban navigation datasets.

Careful wording for supervisor communication:

> This project is inspired by urban navigation and multi-sensor GNSS/INS research directions. It is not a reproduction of a specific paper, but it helped me build a simplified baseline for geometry, robust estimation, and sensor-aiding concepts.

## Prof. Bing Xu-Related Directions

Prof. Xu's research directions include satellite positioning and navigation, GNSS signal processing, LEO PNT, cellular/wireless signal positioning, and GNSS interference or anomaly-related topics.

### Selected Literature-Grounded Conceptual Replication

Selected paper:

> Vicenzo, S., Xu, B., Xu, H., & Hsu, L.-T. (2024). "GNSS direct position estimation-inspired positioning with pseudorange correlogram for urban navigation." GPS Solutions, 28, Article 83. https://doi.org/10.1007/s10291-024-01627-5

Why this paper was selected:

- It is recent and appears on Prof. Xu's publication list.
- It directly addresses urban GNSS multipath/NLOS robustness.
- Its pseudorange-correlogram idea can be reduced to a small, inspectable 2D experiment.
- The paper points to the public UrbanNav dataset for validation data.

Data source boundary:

- Public data used in this MiniLab: UrbanNav-HK-Medium-Urban-1 TST ground-truth route subset.
- Synthetic data used in this MiniLab: satellite geometry, pseudoranges, receiver clock bias, NLOS-like biases, and C/N0 values.
- This is a pseudorange-correlogram-inspired conceptual replication, not an exact reproduction of the original UrbanNav/RINEX experiment.
- UrbanNav is currently used only for the public ground-truth route subset.
- Pseudorange, C/N0, receiver clock bias, satellite geometry, and LOS/multipath/NLOS labels are synthetic.
- No original UrbanNav RINEX pseudorange or real C/N0 is parsed yet.
- No paper table values are manually transcribed and no figure values are digitized.

How it connects to existing modules:

- **DOP/HDOP:** the same simplified source geometry idea remains useful for understanding candidate-position sensitivity.
- **WLS:** the selected paper weights pseudorange consistency by C/N0; the toy implementation uses synthetic C/N0 to connect with the MiniLab's uncertainty-weighted least-squares theme.
- **Robust LS:** the pseudorange-correlogram score downweights or clips inconsistent measurements, which is conceptually related to robust estimation under outliers.
- **GNSS/INS toy fusion:** not directly implemented from the paper, but the same public UrbanNav route could later be used for GNSS/INS-style studies.
- **LEO-like aiding:** not implemented from the paper, though the project can later compare correlogram scoring under added LEO-like ranging geometry.
- **Urban NLOS/multipath simulation:** the conceptual replication injects synthetic multipath/NLOS positive pseudorange bias and lower synthetic C/N0-like weights to mimic the kind of urban degradation discussed by the paper.

High-level ideas reflected in this project:

- **GNSS pseudorange positioning:** the baseline solver estimates 2D position and receiver clock bias from pseudoranges.
- **LEO-like assisted PNT:** the project adds simplified moving ranging sources to study how additional measurements can change HDOP and positioning error.
- **Measurement-level anomaly modeling:** the spoofing-like scenario only adds synthetic pseudorange drift and is used to study residual behavior safely.
- **Robustness under biased measurements:** WLS, Huber IRLS, and residual exclusion provide toy robust-estimation comparisons.
- **Future extension:** the LEO-like module could be replaced with more realistic LEO orbit geometry or signal/ranging assumptions after further study.

Careful wording for supervisor communication:

> This MiniLab helped me start learning GNSS robustness and LEO-aided positioning concepts at the measurement-model level. The current LEO component is only a toy ranging-source abstraction, and I hope to improve it toward more realistic LEO PNT simulation in future work.

## Additional Related Directions

### Urban GNSS Error Modeling and NLOS Classification

The MiniLab uses synthetic LOS/NLOS-style uncertainty labels in the WLS experiment. This is only a simplified label-based uncertainty model. A future version could add geometry-based urban canyon blockage, LOS/NLOS classification, or error-map concepts.

### Measurement Uncertainty Modeling

WLS uses larger standard deviations for measurements labeled as high-uncertainty. This illustrates the basic idea that not all pseudorange measurements should be trusted equally in urban environments. The good WLS result depends on having useful uncertainty or NLOS-label information; without such information, WLS cannot automatically know which measurements to downweight.

### DOP/HDOP and Satellite Geometry

The DOP module computes simplified 2D HDOP from the geometry matrix. This helps connect satellite/source distribution with positioning sensitivity and supports the LEO-like aiding experiment. HDOP only describes geometry; it does not include multipath/NLOS bias, spoofing-like drift, measurement noise, or solver robustness. The clean LEO comparison figures are included to isolate geometry from degraded-measurement effects.

### GNSS/INS Toy Fusion

The project includes a simple Kalman filter with state `[px, py, vx, vy]`. Synthetic velocity/odometry is generated from the simulated ground-truth trajectory with noise, then integrated during prediction. It is a teaching baseline only. A research-grade extension would need full inertial mechanization, IMU error-state modeling, and more rigorous GNSS update handling.

## How This Project Reflects These Ideas

- `src/dop.py`: simplified geometry matrix, HDOP, and GDOP-like metrics.
- `src/gnss_solver.py`: OLS, WLS, Huber IRLS, and residual-exclusion baselines.
- `src/error_models.py`: synthetic multipath/NLOS and spoofing-like pseudorange anomalies.
- `src/urban_environment.py`: geometry-based 2D rectangular-building LOS/NLOS labels, NLOS uncertainty, and urban error maps.
- `src/leo_aiding.py`: LEO-like moving ranging sources for conceptual geometry improvement.
- `src/ins_filter.py`: synthetic odometry, dead reckoning, and a toy GNSS/INS Kalman filter.
- `src/sliding_window.py`: simplified sliding-window least squares with pseudorange residuals and odometry smoothness.
- `src/collaborative_positioning.py`: two-agent collaborative positioning toy model with an inter-agent distance constraint.
- `experiments/run_experiments.py`: fixed metrics and reproducible experiment outputs.
- `experiments/run_all_experiments.py`: one-command pipeline for generated figures, CSV metrics, and reproduction outputs.
- `experiments/plot_results.py`: presentation-friendly figures for supervisor communication.
- `results/clean_leo_hdop_comparison.png` and `results/clean_leo_error_comparison.png`: clean-condition comparison of GNSS-only and GNSS + LEO-like aiding.

## Future Extensions

- Replace simplified geometry with real ephemeris or almanac-based GNSS geometry.
- Add real public urban GNSS datasets.
- Implement 3D ECEF/ENU positioning.
- Add robust factor-graph GNSS/INS optimization.
- Add geometry-based LOS/NLOS urban canyon simulation.
- Add more realistic LEO PNT geometry and ranging assumptions.
- Compare residual-based anomaly detection with more rigorous integrity-monitoring methods.
