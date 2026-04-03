# SpecLens-RL

SpecLens-RL is a framework-first reinforcement learning repository for diagnosing
specification failures at laptop scale. It is not a new RL trick and it is not
a benchmark-SOTA project. The core artifact is a reusable task schema, a metric
suite, leakage-aware monitor evaluation, three symbolic environments, and a
paper scaffold whose empirical sections are populated only from saved outputs.

## Scope

The scientific object in this repo is a specification diagnostic task:

`T = (M, R_p, J_t, C, M_mon, Delta, Pi)`

where:

- `M` is the underlying MDP or POMDP.
- `R_p` is the proxy reward used for learning.
- `J_t` is the hidden true objective used for evaluation.
- `C` is the corruption or tampering channel family.
- `M_mon` is the monitor observation channel.
- `Delta` is the distribution-shift family.
- `Pi` is the policy or training family considered.

This repository packages that framing as code, experiments, and paper assets.

## Environments

The initial benchmark contains three compact symbolic environments:

- `ProxyFarming-v0`: goal-reaching with exploitable local proxy farming.
- `SensorTamper-v0`: delivery with explicit tampering opportunities.
- `GemTimeout-v0`: delayed success versus myopic gem collection.

Each environment exports a schema record, computes proxy and true signals, marks
success and reward-hack events, and supports deterministic ID and OOD variants.

## Methods

The required method set stays intentionally small:

- `PROXY_PPO`: standard PPO trained on proxy reward only.
- `ORACLE_TRUE_PPO`: PPO trained on the true objective for diagnosis only.
- `CONSERVATIVE_PROXY_PPO`: optional and only run when resource-safe.

## Monitor Channels

The repo evaluates monitorability under two explicit audit views:

- `restricted`: observations, actions, reward history, length, and compact
  derived features chosen to avoid direct label leakage.
- `rich`: restricted features plus extra privileged summaries that remain
  plausible in an audit setting without making the label tautological.

Leakage diagnostics are mandatory before any monitor result is accepted.

## Safety Rules

- Never fabricate citations, results, runtime claims, or tables.
- Do not write result sentences into the paper unless supported by saved
  artifacts.
- Stop new runs if free disk falls below `1 GB`.
- Respect the active artifact budget chosen by preflight.
- Keep only compact metrics, sampled trajectories, and at most two checkpoints
  per run.

## Profiles

Preflight selects one of four execution profiles:

- `PROFILE_FULL_SAFE`
- `PROFILE_LEAN_SAFE`
- `PROFILE_ULTRA_LEAN`
- `PROFILE_SCAFFOLD_ONLY`

Expected generated-artifact budgets:

- `FULL_SAFE`: up to `3.0 GB`
- `LEAN_SAFE`: up to `1.5 GB`
- `ULTRA_LEAN`: up to `700 MB`
- `SCAFFOLD_ONLY`: up to `250 MB`

## Commands

```bash
make install
make preflight
make smoke
make stage1
make stage2
make plots
make tables
make paper
make clean_artifacts
make report
```

The stage scripts enforce the active profile and disk budget before launching
new work.

## Paper

The paper scaffold lives in [`latex/`](./latex). Non-experimental sections are
written immediately. Experimental placeholders are only populated after plots
and tables have been generated from real outputs.

## Reproducibility

The repository is fully local. It does not require paid APIs, cloud services,
external databases, or human labeling. Before any future submission cycle, a
human should manually recheck:

- bibliography metadata,
- hardware/runtime statements,
- all populated paper result sentences,
- environment calibration after any override,
- and leakage diagnostics for each accepted monitor result.
