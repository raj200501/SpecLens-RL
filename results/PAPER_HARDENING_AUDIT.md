# Paper Hardening Audit

## Strongest claim the current evidence supports

- The current evidence supports a conservative framework claim rather than a method claim.
- Strongest honest statement:
  - `SpecLens-RL` provides a compact, reproducible framework for measuring specification gaps across three lightweight RL tasks, explicitly measures tampering risk in `SensorTamper-v0`, and supports leakage-controlled monitor evaluation in one environment while showing that monitor labels can degenerate in others.

## Claims that are too strong and should be softened

- Any wording that implies broad monitorability evidence across the full benchmark is too strong.
- Any wording that implies OOD shift uniformly increases hacking is too strong; the current evidence is mixed and environment dependent.
- Any wording that implies the benchmark demonstrates practical oversight for realistic systems is too strong.
- Any wording that centers the repository scaffold or placeholder policy more than the actual empirical contribution should be softened or removed.

## Figures that should remain in the main paper

- A single conceptual framework figure should remain in the main paper.
- A single compact main-results figure should remain in the main paper.
- A dedicated `SensorTamper-v0` monitor figure should remain in the main paper because it carries the only accepted monitor evidence.

## Tables that should remain in the main paper

- Environment instantiation summary.
- Main specification-gap results.
- `SensorTamper-v0` monitor table, if compact and readable.

## Artifacts that should move to appendix, supplement, or be omitted

- Full resource table should leave the main body.
- Full threshold-robustness table should leave the main body.
- The old raw figure set (`proxy_vs_true_scatter`, `hacking_incidence_bars`, `generalization_gap`, `tampering_summary`, `monitorability_gap`) should not all remain in the main flow.
- Probe artifacts in `results/second_pass_probe/` should stay in the repository but should not influence the main-paper presentation.

## Biggest presentation weaknesses

- The results section currently reads like an artifact-status summary, not a scientific argument.
- The experimental setup section still sounds scaffold-oriented and overemphasizes repo mechanics.
- The figures are functional but visually raw and closer to diagnostic plots than publication figures.
- The tables are CSV-like: long raw column names, poor grouping, and too much low-signal information in the main body.
- The paper currently exposes the implementation names (`PROXY_PPO`, `ORACLE_TRUE_PPO`) instead of presentation labels.

## Biggest LaTeX, citation, and formatting weaknesses

- The cite graph contains undefined citation keys.
- The bibliography contains unused seed entries that clutter the file.
- Several sections still use placeholder-style wording (`IfFileExists` flow is fine for the repo, but the visible prose should not sound provisional where artifacts already exist).
- The current main-body artifact count is too high for a realistic seven-page conference presentation.
- No local TeX toolchain is installed in this environment, so page discipline must be enforced structurally rather than by direct compile-time pagination checks.
