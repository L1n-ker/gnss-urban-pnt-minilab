# Limitations

## Technical Limitations

- The positioning model is 2D and local, not full 3D ECEF/ENU GNSS.
- Satellite positions are simplified synthetic geometry unless explicitly stated otherwise.
- Atmospheric delay, satellite clock, Earth rotation, relativity, carrier phase, cycle slips, and real ephemerides are not modeled.
- Urban buildings are simplified 2D rectangles, not real 3D maps.
- LOS/NLOS labels are synthetic and geometry-derived.
- LEO-like sources are conceptual moving range sources, not real LEO orbit or signal models.
- GNSS/INS fusion is a toy constant-velocity Kalman filter, not strapdown INS or an error-state EKF.
- Sliding-window optimization is factor-graph-inspired but not a full factor graph.
- Collaborative positioning is a two-agent relative-distance toy example, not a production cooperative localization system.
- The literature-grounded conceptual replication uses synthetic measurements and should not be described as an exact paper reproduction.

## Experiment Limitations

- Many experiments use known uncertainty labels to demonstrate principles. Real systems need to estimate or classify these uncertainties.
- Some controlled cases show strong improvement because the degradation and uncertainty assumptions are intentionally clear.
- Fixed random seeds make results reproducible but do not prove robustness across all environments.
- Results should be interpreted as conceptual evidence of learning, not real-world benchmark performance.

## Safety Limitations

- The project does not generate, transmit, replay, or decode real GNSS RF signals.
- The project does not include SDR transmission or jamming workflows.
- Spoofing-like behavior is limited to synthetic measurement-array drift inside Python.
- The project cannot be used as a real spoofing, jamming, or receiver attack tool.

## Presentation Boundary

Safe wording:

- "simplified educational implementation"
- "toy demonstration"
- "conceptual simulation"
- "inspired by"
- "research-preparation portfolio"

Avoid wording:

- "real GNSS receiver"
- "real spoofing detection"
- "exact paper reproduction"
- "validated urban-navigation system"
- "proves real-world performance"
