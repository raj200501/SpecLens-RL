# NeurIPS Spec Upgrade Plan

Date: `2026-04-10`

## Chosen improvements

### 1. Add a cross-environment monitor-scope analysis

Why this was chosen:
- This directly addresses the paper's weakest current point.
- It can be done entirely from saved second-pass artifacts.
- It converts "monitor results are narrow" from a disclaimer into a concrete methodological finding.

What success looks like:
- A compact artifact-backed summary of hack-label prevalence, failure-label prevalence, leakage status, and accepted monitor outcomes across the three environments.
- The paper can explain why SensorTamper-v0 yields valid monitor evidence while ProxyFarming-v0 and GemTimeout-v0 remain degenerate under the proxy-only protocol.

How it will be verified:
- Read `results/second_pass/monitor_eval/monitor_datasets/*.csv.gz`.
- Read `results/second_pass/monitor_eval/leakage/*.json`.
- Confirm that any new table/claim matches the saved counts and statuses exactly.

### 2. Run one small SensorTamper-v0 corruption-channel ablation

Why this was chosen:
- It is the cheapest direct test of the paper's strongest environment-specific
  claim.
- It sharpens the explicit corruption-channel story in SensorTamper-v0 without
  launching a new training campaign.

What success looks like:
- Evaluating saved SensorTamper checkpoints with the tamper effect disabled
  should collapse the learned proxy exploit while leaving the oracle policy
  essentially unchanged.
- If the effect is tiny, noisy, or narratively ambiguous, it will stay out of
  the paper.

How it will be verified:
- Add a minimal evaluation-only environment flag:
  `tamper_effect_enabled=False`.
- Evaluate the saved `PROXY_PPO` and `ORACLE_TRUE_PPO` checkpoints with the
  same deterministic ID/OOD protocol used for the benchmark.
- Compare the ablation outputs against the saved baseline summaries using
  generated CSV artifacts.

### 3. Rewrite the paper around the strongest honest identity

Why this was chosen:
- The current paper is already close, but it still reads somewhat like a generic workshop export instead of a deliberately targeted diagnosis/evaluation paper.
- This is high leverage and low risk.

What success looks like:
- The title, abstract, introduction, contributions framing, results interpretation, discussion, and conclusion all clearly support:
  `framework / benchmark paper` first, `evaluation methodology paper` second.
- The monitor story is framed as narrow but meaningful, not as a near-miss broad success.

How it will be verified:
- Every revised claim must appear in `results/SPEC_VERIFY_BEFORE_WRITE.md` with a direct artifact pointer.
- The revised paper should no longer imply broad monitorability support or method-level solving claims.

### 4. Consolidate figures, tables, and support structure

Why this was chosen:
- The main paper should read as a coherent scientific argument, not a repo artifact dump.
- This is the cleanest way to improve workshop readability without changing the science.

What success looks like:
- The main paper emphasizes the framework figure, the main cross-environment results figure, the SensorTamper monitor figure, and only the tables that directly support the top-line claim.
- Support diagnostics are either moved to a compact appendix position or excluded from the main flow.

How it will be verified:
- Re-read the final PDF like a reviewer and check whether each figure/table is doing unique argumentative work.
- Remove or demote anything that is merely decorative or redundant.

## What is intentionally left unchanged

- No broad rerun of the benchmark.
- No new environments.
- No attempt to manufacture nondegenerate monitor tasks in ProxyFarming-v0 or GemTimeout-v0.
- No expansion into a multi-method RL paper.
- No attempt to force the earlier conservative-proxy dead end into the paper.
- No unsupported workshop retargeting to a named official NeurIPS 2026 workshop.

## Filtering rule for this pass

Any improvement that does not clearly sharpen the benchmark/framework story, improve workshop fit, or add directly verifiable evidence will be logged as a trial and excluded from the final paper.

## Addendum after official NeurIPS 2026 ED guidance became public

- The package should no longer optimize around a generic unnamed workshop.
- The paper's primary contribution now aligns more cleanly with the official
  NeurIPS 2026 Evaluations \& Datasets track than with an unspecified workshop:
  benchmark design, evaluation methodology, and artifact-backed empirical
  insight.
- A later hardening pass should therefore treat anonymous ED-track retargeting
  as a valid high-leverage improvement rather than as out-of-scope churn.
