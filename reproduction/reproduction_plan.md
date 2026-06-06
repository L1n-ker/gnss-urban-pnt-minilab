# Pseudorange Correlogram Reproduction Plan

## Objective

Build a small, academically honest simplified reproduction of one component from Vicenzo et al. (2024): pseudorange-correlogram candidate-state scoring for urban GNSS robustness.

## Dataset Or Synthetic Setup

- Public data: a small UrbanNav-HK-Medium-Urban-1 TST ground-truth text file is used as the real trajectory reference.
- Synthetic data: satellite positions, pseudorange measurements, receiver clock bias, LOS/multipath/NLOS labels, multipath/NLOS positive biases, and C/N0 values are generated with fixed random seeds.
- Data label: this is a simplified reproduction / conceptual replication. It does not use original UrbanNav RINEX pseudorange observations, real C/N0, manually transcribed paper table values, digitized figure values, or the paper's generated result files.

## New Files

- `src/pseudorange_correlogram.py`: parser, synthetic case generator, pseudorange-correlogram scoring, optional `[x, y, clock_bias_m]` grid solver, and batch evaluation.
- `tests/test_pseudorange_correlogram.py`: focused tests for parser/scoring/grid robustness.
- `reproduction/run_reproduction.py`: runs the experiment and saves CSV/NPZ/notes.
- `reproduction/plot_reproduction.py`: creates presentation figures from saved arrays.
- `reproduction/literature_search_summary.md`: verified paper candidates and selection rationale.
- `reproduction/selected_paper_summary.md`: one selected-paper summary.
- `reproduction/method_scope.md`: method scope and academic boundary statement.
- `results/reproduction/`: generated metrics, arrays, figures, and notes.

## Existing Files To Update

- `docs/literature_mapping.md`: add the selected paper and module mapping.
- `docs/one_page_report.md`: add the literature-grounded reproduction result.
- `README.md`: add a "Literature-Grounded Reproduction Study" section.

## Algorithm Steps

1. Load a subset of public UrbanNav TST ground-truth positions.
2. Convert latitude/longitude DMS fields into local East-North meters.
3. Generate fixed synthetic 2D satellite positions around the local trajectory.
4. Generate synthetic pseudoranges from geometric range + receiver clock bias + Gaussian noise.
5. Generate two synthetic-measurement scenarios:
   - clean/open-sky-like: LOS only, small noise, high C/N0;
   - NLOS-dominated urban: LOS, multipath, and NLOS measurements with progressively larger bias/noise and lower C/N0.
6. Solve each epoch using the existing OLS time-series solver.
7. For each epoch, run a small grid search around the OLS estimate.
8. Optionally search receiver clock bias in meters as the third state dimension.
9. Score the candidate using a triangular pseudorange consistency function weighted by normalized C/N0.
10. Select the highest-scoring candidate and save metrics.

## Evaluation Metrics

- Mean, median, RMSE, max, and final horizontal positioning error for OLS and pseudorange-correlogram-inspired toy estimates.
- Mean improvement percentage relative to OLS.
- Mean correlogram score.
- Number of multipath and NLOS-like biased measurements.

## Expected Figures

- `reproduced_figure_error_timeseries.png`: OLS vs simplified pseudorange-correlogram positioning error.
- `reproduced_figure_trajectory.png`: public UrbanNav route subset with OLS and correlogram estimates.
- `reproduced_figure_score_map.png`: candidate-position score map for a selected NLOS-heavy epoch.
- `correlogram_xy_slice.png`: XY score slice at the best searched clock-bias value.
- `reproduction_clean_vs_nlos.png`: clean/open-sky-like versus NLOS-dominated scenario comparison.

## How To Run

From the project root:

```bash
python reproduction/run_reproduction.py
```

This downloads or reuses the small public UrbanNav ground-truth file, runs the synthetic pseudorange experiment, and generates all outputs under `results/reproduction/`.

## How To Verify

Run:

```bash
python -m unittest discover -s tests -v
python reproduction/run_reproduction.py
```

Expected behavior:

- tests pass;
- `results/reproduction/reproduction_metrics.csv` exists;
- `results/reproduction/reproduction_scenario_metrics.csv` exists;
- all requested figure PNG files exist;
- `results/reproduction/reproduction_notes.md` states the public/synthetic data boundary.

## Limitations

This does not reproduce the paper's exact UrbanNav RINEX-based results. It only implements one algorithmic idea at the 2D synthetic pseudorange-measurement level. It should be presented as "inspired by", "simplified reproduction", "conceptual replication", or "pseudorange-correlogram-inspired toy implementation", not as an exact reproduction of the original system.
