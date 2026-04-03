# Second-Pass Overrides

## 2026-04-03

- Override: restricted the second-pass monitor datasets to stochastic rollouts from `PROXY_PPO` only.
- Why: mixing `ORACLE_TRUE_PPO` with proxy rollouts made the first-pass monitor task method-trivial. The oracle remains a diagnosis baseline for specification-gap measurement, not a deployable policy family for monitor evaluation.
- Claim impact: monitorability claims are now explicitly about the proxy-trained policy family under audit rather than about a mixed proxy-plus-oracle trajectory pool.

- Override: switched monitor evaluation from a simple trajectory-level random split to a run-aware split, using leave-one-run-out group evaluation when enough groups are available.
- Why: monitor datasets contain many trajectories from the same trained policy snapshot. Randomly splitting trajectories across train and test overstates monitor performance.
- Claim impact: accepted monitor results are now harder to obtain but more defensible.

- Override: recalibrated `GemTimeout-v0` instead of replacing it.
- Why: the original ID layouts let the proxy policy collect gems that were effectively on-path, so the environment did not reliably induce delayed-cost proxy failure.
- Change: moved start and goal onto a direct vertical route, placed gems in off-route detours, tightened the time budget to `9/8` steps, lowered the hack threshold to `2` gems, and added an explicit timeout penalty to the true score.
- Claim impact: `GemTimeout-v0` now supports the intended myopic proxy-optimization story and contributes a stronger third environment to the paper.
