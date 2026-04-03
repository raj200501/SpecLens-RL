# Second-Pass Final Report

## What changed

- Monitor datasets were rebuilt from stochastic `PROXY_PPO` rollouts only.
- Monitor evaluation now uses run-aware splitting instead of a simple random trajectory split when possible.
- `GemTimeout-v0` was recalibrated rather than replaced.
- Second-pass outputs were written under `results/second_pass/`, with top-level aggregate exports copied to:
  - `results/second_pass_aggregated_main_results.csv`
  - `results/second_pass_aggregated_monitor_results.csv`
  - `results/second_pass_aggregated_tampering_results.csv`
  - `results/second_pass_aggregated_resource_results.csv`

## Why it changed

- The first-pass monitor task mixed proxy and oracle trajectories, which made monitor prediction method-trivial.
- The first-pass monitor split was too permissive because trajectories from the same checkpoint could land on both sides of the split.
- The first-pass `GemTimeout-v0` did not consistently induce delayed-cost proxy failure in-distribution.

## What actually reran

- `ProxyFarming-v0`: deterministic reevaluation from saved checkpoints, plus proxy-only stochastic monitor rollouts.
- `SensorTamper-v0`: deterministic reevaluation from saved checkpoints, plus proxy-only stochastic monitor rollouts.
- `GemTimeout-v0`: retrained for `60000` steps per run for `PROXY_PPO` and `ORACLE_TRUE_PPO`, then reevaluated deterministically and with proxy-only stochastic monitor rollouts.

## Monitorability status

- Valid leakage-controlled hack-monitor results now exist for `SensorTamper-v0`.
- `SensorTamper-v0` restricted-channel hack AUCs:
  - logistic regression: `ROC-AUC 0.936`, `PR-AUC 0.962`
  - gradient boosting: `ROC-AUC 0.964`, `PR-AUC 0.984`
- `SensorTamper-v0` rich-channel hack AUCs:
  - logistic regression: `ROC-AUC 0.964`, `PR-AUC 0.986`
  - gradient boosting: `ROC-AUC 0.965`, `PR-AUC 0.984`
- The rich-minus-restricted gap is positive but modest, not a dramatic jump.
- `ProxyFarming-v0` and `GemTimeout-v0` monitor targets remain degenerate under the proxy-only ID monitor protocol.

## GemTimeout status

- `GemTimeout-v0` was fixed, not replaced.
- The second-pass environment now produces a clear third benchmark story:
  - `ORACLE_TRUE_PPO`: `true_score_ID 1.000`, `hacking_ID 0.000`
  - `PROXY_PPO`: `true_score_ID -0.153`, `hacking_ID 1.000`
- This now complements `ProxyFarming-v0` and `SensorTamper-v0` rather than weakening the empirical section.

## Final paper story

- Story A is now honestly supportable with qualification.
- Supported claim:
  - SpecLens-RL measures specification gaps across all three environments.
  - It measures explicit tampering risk in `SensorTamper-v0`.
  - It supports at least one leakage-controlled monitorability comparison in `SensorTamper-v0`.
- Required qualification:
  - monitor evidence is concentrated in one environment
  - proxy-only monitor labels are still degenerate in `ProxyFarming-v0` and `GemTimeout-v0`

## What remains weak

- The monitor story is no longer collapsed, but it is still narrow.
- `ProxyFarming-v0` is empirically strong for specification-gap measurement but weak for nondegenerate proxy-only monitor evaluation because the saved proxy policy hacks almost all ID trajectories.
- `GemTimeout-v0` is now strong as a specification-gap environment, but its proxy-only monitor labels are also degenerate.
- The second pass did not add the optional corruption on/off ablation.

## Leakage summary

- Leakage diagnostics under the second-pass monitor protocol:
  - `pass`: `4`
  - `degenerate`: `8`
- No accepted second-pass monitor row is currently marked `warning` or `fail`.

## Runtime and disk

- Second-pass runner wall clock: `102.223` seconds.
- `results/second_pass/` disk usage: about `8.6 MB`.

## Exact reproduction commands

- `./.venv/bin/pytest -q`
- `./.venv/bin/python scripts/run_second_pass.py`

## Human follow-up before submission

- Recheck whether the paper should emphasize the valid `SensorTamper-v0` monitor result or frame the monitor section more conservatively around concentration in a single environment.
- Review whether the paper text should explicitly note that monitor datasets use proxy-only stochastic rollouts while main task evaluation remains deterministic.
- Re-verify bibliography metadata and final LaTeX formatting before any submission cycle.
