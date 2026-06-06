# Literature Search Summary

This search focused on recent representative work by Prof. Bing Xu in GNSS, urban navigation, multipath/NLOS mitigation, robust positioning, and LEO PNT. Prof. Xu's official PolyU profile identifies his specialization as satellite positioning and navigation, GNSS signal processing, LEO PNT, cellular positioning, GNSS radio-frequency interference detection/localization, and wireless communications: <https://www.polyu.edu.hk/aae/people/academic-staff/dr-xu-bing/>. His lab publication page was used as the primary identity check: <https://pbingxu.github.io/publications/>.

## Candidate Papers

| Candidate paper | Verification source | Data/code availability observed | Main method | Difficulty | Relevance | Possible component |
|---|---|---|---|---|---|---|
| Vicenzo, S., Xu, B., Xu, H., & Hsu, L.-T. (2024). "GNSS direct position estimation-inspired positioning with pseudorange correlogram for urban navigation." GPS Solutions, 28, 83. DOI: 10.1007/s10291-024-01627-5 | PolyU IRA and Springer page list title, authors, year, venue, DOI, abstract, and open-access record. | The paper states that UrbanNav datasets used for validation are open-source at `https://github.com/IPNL-POLYU/UrbanNavDataset`; generated results are available from the corresponding author on request. No public implementation of the selected paper was found during this search. | Grid-based MLE-like pseudorange correlogram that scores candidate states using pseudorange consistency and C/N0 weighting. | Medium | High | Pseudorange-correlogram-inspired toy implementation compared with OLS on a public UrbanNav ground-truth route subset plus synthetic pseudorange/C/N0 measurements. |
| Qi, X., Xu, B., Wang, Z., & Hsu, L.-T. (2024). "Random forest-based multipath parameter estimation." GPS Solutions, 28, 126. DOI: 10.1007/s10291-024-01667-x | Prof. Xu lab publications and Springer page. | Open-access article; public training dataset/code not found in this quick search. | Random forest estimates multipath parameters from correlation-domain features. | High | Medium | Synthetic RF-style feature regression, but this is less aligned with the current 2D pseudorange simulator. |
| Xu, B., Jia, Q., & Hsu, L.-T. (2020). "Vector Tracking Loop-Based GNSS NLOS Detection and Correction." IEEE Transactions on Instrumentation and Measurement, 69(7), 4604-4619. | Prof. Xu lab publications and PolyU Scholars Hub. | Public article metadata found; no small public dataset/code confirmed here. | Vector tracking loop metrics for NLOS detection/correction. | High | High | Measurement-level NLOS detection idea, but full VTL reproduction would require SDR/tracking-loop infrastructure. |
| Xu, B., & Hsu, L.-T. (2019). "Open-source MATLAB code for GPS vector tracking on a software-defined receiver." GPS Solutions, 23, 46. DOI: 10.1007/s10291-019-0839-x | PolyU Scholars Hub and GPS Toolbox listing. | Public MATLAB SDR/vector tracking code is the focus of the paper. | GPS vector tracking SDR with EKF navigation processor. | High | Medium | EKF/vector tracking concept, but it is older, MATLAB/SDR-based, and outside this Python project's measurement-level scope. |
| Zhang, Q., & Xu, B. (2025). "Analysis of multipath effects on LEO ranging-based positioning using BPSK and BOC signals in urban areas." Advances in Space Research. DOI: 10.1016/j.asr.2024.11.049 | Prof. Xu lab publications and publisher/search metadata. | Public code/data not found in this quick search. | LEO ranging signal multipath analysis for BPSK/BOC. | Medium-high | High | Synthetic LEO multipath geometry study, but the signal-level assumptions are less beginner-friendly than the pseudorange-correlogram target. |

## Selected Target

The selected target is the 2024 GPS Solutions paper on pseudorange correlogram positioning. It is recent, directly related to urban GNSS robustness, and naturally maps to the MiniLab's existing pseudorange solver.

## Data Source Decision

Data priority was applied as follows:

1. Public code from the selected paper: not found in this search.
2. Public dataset used by the paper: partially used. The paper points to the open-source UrbanNav dataset, and the UrbanNav repository lists downloadable GNSS RINEX and ground-truth files. This conceptual replication uses only a small public UrbanNav-HK-Medium-Urban-1 TST ground-truth text file as the real route reference.
3. Reported paper table values: not used as numerical inputs.
4. Digitized figure values: not used.
5. Synthetic data: used for satellite geometry, pseudorange, C/N0, receiver clock bias, LOS/multipath/NLOS labels, and multipath/NLOS biases because this MiniLab does not yet parse the full RINEX/RTKLIB workflow. These synthetic measurements are clearly labeled as synthetic; they are not original raw data from the paper.

Therefore, the output is labeled as a conceptual replication of one algorithmic idea, not an exact reproduction of the paper's UrbanNav/RINEX experiments.
