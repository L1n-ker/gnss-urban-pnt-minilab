# Selected Results Captions

These captions are written for a supervisor-facing GitHub README or short project update.

**Figure 1. Controlled robust solver comparison.** OLS, WLS, Huber IRLS, Cauchy IRLS, and residual exclusion are compared under synthetic NLOS / multipath-like pseudorange bias. The purpose is to show how weighting and robust estimation can reduce sensitivity to biased pseudorange measurements in a simplified positioning problem.

**Figure 2. Synthetic urban LOS/NLOS demonstration.** A simplified 2D urban-canyon model uses rectangular building blocks to label receiver-to-source links as LOS or NLOS. NLOS measurements are assigned larger uncertainty and positive range bias in later experiments.

**Figure 3. GNSS + LEO-like aiding under degraded measurements.** The experiment compares GNSS-only and GNSS + LEO-like positioning when synthetic NLOS / multipath-like errors are present. It illustrates an important point: improved geometry does not automatically guarantee lower positioning error if biased measurements remain unmitigated.

**Figure 4. Pseudorange-correlogram-inspired candidate score map.** A toy candidate-state search evaluates pseudorange consistency and synthetic C/N0-like weighting over a 2D grid. The goal is to understand the intuition behind direct-position-estimation-style candidate scoring, not to reproduce a real RINEX/IF-based experiment.

**Figure 5. OLS vs pseudorange-correlogram-inspired toy method.** Under a fixed synthetic NLOS-dominated scenario, the candidate-scoring method is compared with an OLS pseudorange solution over time. The result provides a compact view of how a consistency-based search can behave differently from direct least-squares fitting.
