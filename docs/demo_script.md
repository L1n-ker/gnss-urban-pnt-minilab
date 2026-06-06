# Demo Script

## 3-Minute Supervisor Demo

1. Open the GUI with:

```text
python app.py
```

2. Start with the project identity:

> This is my GNSS Urban PNT MiniLab. I built it as a research-preparation portfolio to connect GNSS pseudorange positioning, urban degradation, robust estimation, assisted navigation, and one simplified literature-inspired reproduction.

3. Show the interactive tabs:

> The first tabs are interactive. I can change measurement noise, NLOS probability, LEO-like aiding, and INS aiding, then inspect trajectory, error, residuals, source geometry, and metrics.

4. Show the research-result tabs:

> The later tabs are fixed reproducible experiments loaded from `results/`. They include DOP/HDOP, robust LS, urban LOS/NLOS, GNSS/INS toy fusion, LEO-like aiding, sliding-window optimization, urban error maps, collaborative positioning, and pseudorange-correlogram reproduction.

5. Emphasize academic honesty:

> Everything is simplified. This is not a real receiver, not SDR, not real spoofing detection, and not a full paper reproduction. The value is that I started building the modeling, coding, experiment, and literature-connection foundation.

## 7-Minute Supervisor Demo

1. Motivation:

> I wanted to prepare for GNSS/urban navigation and assisted PNT research by building a small but reproducible Python project. My focus was not only to run a simulation, but to connect each module to a research idea and generate figures and metrics.

2. Robust GNSS line:

> The baseline estimates 2D position and receiver clock bias from pseudoranges. I added synthetic multipath/NLOS bias, residual diagnostics, OLS, WLS, Huber IRLS, and residual exclusion. The controlled solver comparison shows why WLS can work well when LOS/NLOS uncertainty labels are meaningful.

3. Urban model:

> I added a simplified 2D urban environment. Rectangular buildings block receiver-to-source lines, creating LOS/NLOS labels. NLOS measurements get positive range bias and larger sigma. The urban error map shows how error varies over a grid.

4. Assisted PNT line:

> For GNSS/INS, I used synthetic velocity from the simulated trajectory, then compared GNSS-only, odometry-only, and a constant-velocity Kalman filter. For LEO-like aiding, I used abstract moving ranging sources to study geometry and HDOP, while clearly not claiming real LEO PNT.

5. Optimization and collaboration:

> I added a sliding-window least-squares toy optimizer inspired by factor graphs. It combines pseudorange residuals with odometry smoothness constraints. I also added a two-agent collaborative positioning toy demo with a relative distance constraint.

6. Literature-grounded reproduction:

> I selected a recent GPS Solutions paper involving Prof. Bing Xu on pseudorange correlogram positioning. My reproduction is intentionally limited: it uses a public UrbanNav route subset, but synthetic pseudorange and C/N0 measurements. It demonstrates one algorithmic idea rather than reproducing the full paper.

7. Close:

> My next step would be to improve realism: 3D ENU/ECEF positioning, real ephemeris or RINEX/RTKLIB-derived measurements, a better GNSS/INS EKF, and more rigorous comparison on public urban navigation datasets.
