# Method Notes

## Pseudorange Model

The MiniLab uses:

```text
pseudorange = geometric range + receiver clock bias + measurement error
```

The estimated state is `[x, y, clock_bias_m]`. The model is 2D and local, not full 3D ECEF/ENU GNSS positioning.

## OLS, WLS, Robust LS, And Residual Exclusion

OLS is the baseline. WLS uses measurement standard deviations and weights measurements by `1 / sigma^2`. Huber IRLS and Cauchy IRLS are included as generic robust-estimation baselines. In the urban toy model, LOS measurements receive small sigma and NLOS measurements receive larger sigma.

Why WLS may outperform generic robust LS when LOS/NLOS uncertainty labels are available:

- WLS receives useful prior information about which measurements are less reliable.
- Generic robust LS only sees residual size after solving; if the initial solution is pulled by biased measurements, residuals can be misleading.
- Residual exclusion depends strongly on the chosen threshold and may remove too much or too little.

This is a teaching comparison, not a certified integrity-monitoring algorithm.

## Urban LOS/NLOS Toy Model

The urban model uses 2D rectangular buildings. A measurement is labeled NLOS if the receiver-to-source line segment intersects a building rectangle. NLOS measurements receive positive pseudorange bias, larger variance, and lower WLS weight.

This is not a full 3D mapping-aided GNSS model. It does not model elevation, diffraction, reflection physics, material properties, or real city maps.

## GNSS/INS Toy Fusion

Synthetic odometry is generated from the simulated ground-truth trajectory with noise and drift. Dead reckoning integrates the noisy velocity. The Kalman filter uses a constant-velocity state `[px, py, vx, vy]` and GNSS position updates.

This is not strapdown INS mechanization and not an error-state GNSS/INS EKF.

## LEO-Like Aiding

LEO-like sources are abstract moving ranging beacons. They are useful for studying additional observation geometry and HDOP changes.

They are not real LEO satellite orbits, real LEO PNT signals, Doppler models, or ephemeris-based predictions.

## Sliding-Window Least Squares

The sliding-window module estimates `[x, y, clock_bias]` for several recent epochs. It combines pseudorange residuals with synthetic odometry smoothness constraints and a first-state prior.

This is factor-graph-inspired but not a complete factor graph implementation.

## Collaborative Positioning

The collaborative module jointly estimates two agents using each agent's GNSS pseudoranges and one relative distance measurement. It demonstrates how a relative constraint can stabilize a degraded individual solution in a controlled toy case.

This is not a full cooperative localization, V2X, UWB, or networked positioning system.

## Literature-Grounded Reproduction

The pseudorange-correlogram reproduction is inspired by Vicenzo, Xu, Xu, and Hsu (2024). It uses a public UrbanNav route subset and synthetic pseudorange/C/N0 measurements.

It is not a full reproduction of the paper's raw-data pipeline or reported results.
