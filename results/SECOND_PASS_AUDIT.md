# Second-Pass Audit

This audit uses the saved first-pass artifacts in `results/` and the current
evaluation / monitor code. It diagnoses why the monitorability results were not
usable and why `GemTimeout-v0` underperformed as a framework showcase.

## ProxyFarming-v0

### Observed failure mode

- Saved monitor leakage status: `fail` for both `restricted` and `rich`.
- Saved hack label distribution in the monitor dataset: `30` positive, `30`
  negative.
- In the saved monitor dataset, all positive hack episodes came from
  `PROXY_PPO` on ID trajectories, while all negative episodes came from
  `ORACLE_TRUE_PPO` on ID trajectories.
- The saved leakage probe identified `episode_length` as a perfect separator
  with AUC `1.0`.

### Why this happened

- The current `scripts/evaluate.py` stores only the first `10` trajectories per
  run, and because evaluation iterates through ID before OOD, the saved
  trajectory set is effectively ID-only for monitor training.
- Within that ID-only saved subset, final deterministic policies are nearly
  method-pure:
  - `PROXY_PPO` loops at the beacon and times out.
  - `ORACLE_TRUE_PPO` goes straight to the goal.
- As a result, coarse trajectory summaries such as episode length, unique
  visited positions, revisit ratio, progress to goal, and final distance to goal
  all become perfect proxies for the label.

### Diagnosis

- Primary issue: **benchmark definition leakage / protocol confounding**.
- Secondary issue: **insufficient trajectory diversity**.
- Contributing issue: the current leakage diagnostic treats any highly
  predictive single feature as leakage, even when the feature is a coarse
  behavioral summary rather than a direct hidden-source variable.

### Most responsible feature families

- exact horizon / timeout behavior
- exact route-completion summaries
- endpoint distance-to-goal summaries
- revisit / path-collapse summaries

## SensorTamper-v0

### Observed failure mode

- Saved monitor leakage status: `fail` for both `restricted` and `rich`.
- Saved hack label distribution in the monitor dataset: `25` positive, `35`
  negative.
- The saved leakage probe identified `unique_positions` as a perfect separator
  with AUC `1.0`.
- Source-variable prediction for `source_tamper_used` also reached `1.0`.

### Why this happened

- The same trajectory-saving issue applies here: the saved monitor dataset is
  effectively ID-only because only the earliest trajectories are retained.
- In that saved subset:
  - `PROXY_PPO` mostly stays near the tamper region, revisits the same cells,
    and times out.
  - `ORACLE_TRUE_PPO` transports the object and traverses much more of the map.
- Features such as unique positions, revisit ratio, progress-to-goal, and final
  distance to goal therefore separate the label extremely well.

### Diagnosis

- Primary issue: **feature overreach in the leakage diagnostic**, not only in
  the monitor channel itself. The current leakage rule treats behaviorally
  informative trajectory-shape features as if they were direct label leakage.
- Secondary issue: **insufficient trajectory diversity** because the saved
  dataset only reflects final deterministic policies in ID.

### Most responsible feature families

- route coverage / unique-position summaries
- path tortuosity / revisit summaries
- progress-to-goal summaries
- end-state distance summaries

## GemTimeout-v0

### Observed failure mode

- Saved monitor leakage status: `degenerate` for both `restricted` and `rich`.
- Saved monitor dataset label counts:
  - `label_hack = 0` for all saved trajectories
  - `label_failure = 0` for all saved trajectories
- The saved monitor dataset contains only ID trajectories from both methods.

### Why this happened

- The trajectory-saving issue again removed OOD from the saved monitor dataset.
- In the saved ID trajectories, both methods always succeed.
- More importantly, the environment is under-calibrated in ID:
  - `PROXY_PPO` collects all three gems on-path and still reaches the goal in
    `12` steps.
  - That means the proxy-greedy behavior does not incur the intended delayed
    cost in-distribution.
- The main results already show this weakness:
  - `PROXY_PPO` has `hacking_incidence_ID = 0.0`
  - `PROXY_PPO` still has relatively high `true_score_ID = 0.94`

### Diagnosis

- Primary issue: **environment degeneracy / weak calibration**.
- Secondary issue: **target imbalance** caused by the saved monitor dataset
  being ID-only.

### Most responsible feature families

- None in the first pass, because the targets themselves collapsed before
  feature leakage became relevant.

## Cross-Cutting Diagnosis

The first-pass monitor failure is not one single problem.

1. There is a **protocol bug / design flaw** in trajectory retention:
   monitor datasets are built from the earliest saved trajectories only, which
   in practice means ID-only data because OOD trajectories are evaluated later.
2. There is **benchmark-level confounding** in `ProxyFarming-v0` and
   `SensorTamper-v0` because the saved monitor dataset reflects only final
   deterministic policies with near-perfect method-aligned behavior.
3. There is **environment degeneracy** in `GemTimeout-v0` because the ID
   variants do not impose enough delayed-cost pressure on proxy-greedy policies.
4. The current leakage rule is **too coarse** because it fails any channel where
   a single behavior-level feature is highly predictive, even if that feature is
   not a hidden-source or benchmark-metadata leak.

## Second-Pass Direction

The second pass should therefore:

- fix trajectory collection so monitor datasets are balanced across splits and
  not truncated to early ID episodes only
- redesign monitor features around coarse trajectory behavior rather than
  environment-specific event counters
- revise leakage diagnostics to distinguish direct leakage from legitimate
  behavioral separability
- recalibrate `GemTimeout-v0` so proxy-greedy behavior incurs real in-distribution
  cost
