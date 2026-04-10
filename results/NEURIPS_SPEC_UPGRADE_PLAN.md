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

### 2. Run one small SensorTamper-v0 conservative-proxy ablation

Why this was chosen:
- It is the cheapest plausible way to test whether the benchmark distinguishes a non-oracle policy family in a reviewer-visible way.
- It is directly aligned with the explicit corruption-channel story in SensorTamper-v0.

What success looks like:
- Relative to Proxy PPO, the conservative proxy policy should reduce tampering or hack incidence and improve true score in a stable enough way to be worth mentioning.
- If the result is unstable, tiny, or narratively confusing, it will stay out of the paper.

How it will be verified:
- Train `CONSERVATIVE_PROXY_PPO` on `SensorTamper-v0` for the existing step budget.
- Evaluate it with the same deterministic ID/OOD protocol used for the main benchmark.
- Compare against the saved Proxy PPO and Oracle True PPO results using generated CSV artifacts.

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
- No unsupported workshop retargeting to a named official NeurIPS 2026 workshop.

## Filtering rule for this pass

Any improvement that does not clearly sharpen the benchmark/framework story, improve workshop fit, or add directly verifiable evidence will be logged as a trial and excluded from the final paper.
