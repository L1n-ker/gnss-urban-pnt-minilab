# Method Scope

## What Is Reproduced

This project implements a simplified reproduction / conceptual replication of one algorithmic idea from Vicenzo, Xu, Xu, and Hsu (2024): pseudorange-correlogram-inspired candidate-state scoring with C/N0 weighting.

The implemented toy candidate state is:

```text
[x, y, clock_bias_m]
```

where `clock_bias_m` is receiver clock bias expressed in meters, consistent with the MiniLab's simplified pseudorange model.

## What Is Simplified

- The solver uses 2D local East-North coordinates, not full 3D ECEF or ENU positioning.
- The search uses a small local grid around an OLS estimate.
- The XY heatmap is a slice through the best searched clock-bias value.
- Satellite geometry, pseudorange measurements, C/N0 values, LOS/multipath/NLOS states, and receiver clock bias are synthetically generated.
- The synthetic measurement model assumes:
  - LOS: small zero-mean pseudorange noise and high C/N0.
  - Multipath: moderate positive bias/noise and moderate C/N0.
  - NLOS: larger positive bias/noise and lower C/N0.

## What Is Not Reproduced

- The original UrbanNav/RINEX processing workflow from the paper.
- Real UrbanNav pseudorange measurements.
- Real UrbanNav C/N0 values.
- Real satellite ephemerides.
- Full 3D positioning.
- IF-level direct position estimation.
- The paper's exact figures, tables, or generated result files.

## Data Provenance

- Public data used: UrbanNav-HK-Medium-Urban-1 TST ground-truth route subset.
- Public data not used yet: UrbanNav RINEX pseudorange, real C/N0, and skymask labels.
- Synthetic data used: satellite positions, pseudorange measurements, receiver clock bias, C/N0, LOS/multipath/NLOS labels, multipath bias, and NLOS positive bias.
- No numerical values from paper tables are manually transcribed.
- No values from paper figures are digitized.

## Why This Still Demonstrates Understanding

The selected paper's key positioning idea is that a candidate navigation state can be scored directly in the navigation domain by comparing candidate-satellite ranges with received pseudoranges, while weaker or less reliable measurements contribute less through C/N0 weighting. This toy implementation demonstrates that idea in a small, inspectable setting:

- in a clean/open-sky-like synthetic setup, OLS and the correlogram toy behave similarly;
- in an NLOS-dominated synthetic setup, the C/N0-weighted correlogram toy is less affected by biased, low-C/N0 measurements than OLS;
- the score-slice figure makes the candidate-state scoring logic visible.

## Next Step Toward Closer Reproduction

The next academically meaningful step is to parse UrbanNav RINEX pseudorange and C/N0 values and connect them with skymask/visibility labels. That would move this from a synthetic-measurement conceptual replication toward a closer reproduction of the original UrbanNav/RINEX experiment.
