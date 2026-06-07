# Selected Paper Summary

## Paper

Vicenzo, S., Xu, B., Xu, H., & Hsu, L.-T. (2024). "GNSS direct position estimation-inspired positioning with pseudorange correlogram for urban navigation." GPS Solutions, 28, Article 83. DOI: <https://doi.org/10.1007/s10291-024-01627-5>

## Verified Source Links

- PolyU Institutional Research Archive record: <https://ira.lib.polyu.edu.hk/handle/10397/104958>
- Springer article page: <https://link.springer.com/article/10.1007/s10291-024-01627-5>
- Author publication page: <https://pbingxu.github.io/publications/>
- UrbanNav dataset repository cited by the paper: <https://github.com/IPNL-POLYU/UrbanNavDataset>

## Research Problem

Urban GNSS positioning suffers from multipath and NLOS pseudorange errors. Conventional two-step positioning can be pulled away from the true position by biased measurements. The paper proposes a lower-complexity method inspired by direct position estimation (DPE): instead of correlating raw IF signals, it builds a pseudorange correlogram over candidate navigation states.

## Method Summary

The paper scores each candidate position by comparing candidate-satellite geometric ranges with received pseudoranges. It uses a triangular code-phase consistency score and weights satellite contributions by carrier-to-noise ratio. This allows highly inconsistent or weak measurements to contribute less to the final candidate score.

## Simplified Component Implemented

This MiniLab implements a simplified 2D synthetic-measurement version of the pseudorange-correlogram idea:

- use a grid of candidate `[x, y, clock_bias_m]` states around an initial OLS solution;
- express receiver clock bias in meters;
- generate an XY score heatmap slice at the best searched clock-bias value;
- compute a triangular pseudorange consistency score inspired by Equation (6) of the paper;
- weight each satellite by normalized C/N0;
- select the candidate with the highest aggregate score;
- compare OLS and the pseudorange-correlogram-inspired toy implementation under clean/open-sky-like and NLOS-dominated synthetic-measurement scenarios.

## Data Provenance

The route comes from a public UrbanNav-HK-Medium-Urban-1 TST ground-truth text file downloaded from the UrbanNav dataset links. The pseudorange measurements, satellite geometry, receiver clock bias, LOS/multipath/NLOS labels, multipath/NLOS biases, and C/N0 values are synthetic and generated inside this project with fixed random seeds. They are not original raw measurements from the paper. The code does not parse UrbanNav RINEX pseudorange or real C/N0 values.

No paper table values are manually transcribed, and no figure values are digitized.

## Simplifications

- 2D local East-North coordinates rather than full 3D ECEF positioning.
- Synthetic GNSS satellite geometry rather than real ephemerides.
- Synthetic pseudorange/C/N0 values rather than RINEX-derived measurements.
- Small local grid search rather than the paper's full workflow and real UrbanNav positioning pipeline.
- No SDR, IF-signal processing, RTKLIB processing, or hardware receiver workflow.

## What I Learned

This component connects the MiniLab's existing OLS/WLS/robust-LS work to a recent pseudorange-correlogram paper. It shows how an algorithm can reduce the influence of inconsistent, low-C/N0 NLOS-like measurements by scoring candidate states directly, while also making clear that this is only a pseudorange-correlogram-inspired toy implementation, not an exact reproduction of the original UrbanNav/RINEX experiment.
