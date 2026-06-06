# GNSS Urban PNT MiniLab - Urban Navigation and Robust PNT Update

This note presents the same MiniLab from an urban navigation and robust PNT perspective. The project is a learning portfolio rather than a finished research system. Its value is that it connects basic pseudorange positioning with several controlled urban-degradation and assisted-navigation experiments that can be inspected, regenerated, and extended.

## Motivation

Urban GNSS positioning is challenging because measurement quality can vary sharply with geometry, blockage, multipath, and NLOS propagation. I built this MiniLab to study those effects in a simplified setting before moving toward real public datasets and more rigorous urban navigation methods. The current project is deliberately transparent: the measurement models, random seeds, solver choices, and limitations are visible in the code and documentation.

## Urban LOS/NLOS Toy Model

The project includes a simplified 2D urban-canyon model in `src/urban_environment.py`. Rectangular building blocks are used to classify receiver-to-source line segments as LOS or NLOS. This does not represent a full 3D map-matching or mapping-aided GNSS system. Its purpose is to create controlled LOS/NLOS labels and measurement uncertainty so that robust positioning methods can be compared under repeatable urban-like degradation.

The selected result `results/selected/02_urban_los_nlos_demo.png` shows the toy buildings, receiver trajectory, and LOS/NLOS behavior over time.

## Urban Error Map

The urban error-map experiment evaluates positioning error over a 2D grid with synthetic blockage and NLOS bias. It compares OLS and WLS under controlled uncertainty assumptions. The output is intended to show how spatial geometry and NLOS weighting can affect positioning quality across an urban-like scene. It should be read as a diagnostic visualization, not as a real city-scale benchmark.

## WLS and Robust LS Under NLOS

The solver module implements OLS, WLS, Huber IRLS, Cauchy IRLS, and residual-exclusion baselines. Under synthetic NLOS / multipath-like pseudorange bias, these methods help me study how measurement weighting, robust losses, and residual thresholds influence the final position estimate. The result is not a complete integrity-monitoring system such as RAIM or ARAIM. It is a controlled educational comparison of robust least-squares behavior.

## GNSS/INS Toy Fusion

The GNSS/INS component uses synthetic velocity / odometry measurements and a toy constant-velocity Kalman filter. It demonstrates the idea that inertial or odometry-like aiding can smooth short degraded intervals, but it is not a strapdown INS mechanization, not an error-state EKF, and not a production navigation filter. The current implementation is useful for understanding the role of motion priors before studying more realistic GNSS/INS integration.

## Sliding-Window Least Squares

The sliding-window module is inspired by factor-graph ideas: it estimates a short sequence of states using pseudorange residuals, a first-state prior, and synthetic odometry smoothness constraints. I avoid calling it a full factor graph because it does not use a mature graph-optimization framework, marginalization strategy, or full sensor-factor design. It is a compact way to understand how temporal constraints can stabilize estimates during degraded GNSS intervals.

## Collaborative Positioning Toy Demo

The collaborative-positioning module uses a two-agent toy example with GNSS pseudorange observations and one inter-agent relative-distance constraint. This demonstrates how a relative constraint can help a degraded agent in a controlled simulation. It is not a full V2X, UWB, cooperative SLAM, or production collaborative-localization system.

## Current Learning Value and Next Steps

The current project shows that I have started building the foundations for urban navigation research: pseudorange positioning, measurement uncertainty, robust estimation, geometry analysis, synthetic NLOS modeling, simple aiding, and reproducible evaluation. The next steps are to move from toy data toward real public urban GNSS datasets, add real RINEX pseudorange and C/N0 ingestion, improve coordinate and satellite-geometry modeling, and compare robust LS with more rigorous GNSS/INS or graph-optimization methods.
