# Override Log

## 2026-04-03

- Override: removed reward-history aggregate features from the monitor channel specs.
- Why: Stage 0 leakage diagnostics on `ProxyFarming-v0` showed that features such as `reward_mean` trivially predicted the hack label, making monitorability results scientifically weak.
- Claim impact: monitorability claims are now narrower. They characterize detection from behavior and compact progress summaries rather than from direct reward-history aggregates.

- Override: removed benchmark split indicators and risky-zone summary counts from the monitor channels.
- Why: after the first monitor redesign, `split_is_ood` and risky-zone summaries still trivialized hack prediction in Stage 0.
- Claim impact: monitor results now focus on trajectory structure and end-state summaries rather than explicit benchmark metadata or near-label source variables.

## 2026-04-03 Second Pass

- Override: second-pass monitor datasets are built from stochastic `PROXY_PPO` rollouts only, not from mixed proxy and oracle trajectories.
- Why: the oracle is a diagnosis baseline rather than a deployable policy family, and the mixed first-pass dataset made monitor prediction largely method-trivial.
- Claim impact: accepted monitorability claims now apply to the proxy-trained policy family under audit.

- Override: second-pass monitor evaluation uses run-aware splitting, with leave-one-run-out evaluation when enough groups are available.
- Why: random trajectory splits can leak policy-specific idiosyncrasies across train and test when multiple trajectories come from the same checkpoint.
- Claim impact: reported monitor AUCs are more conservative and more defensible.

- Override: `GemTimeout-v0` was recalibrated with off-route gem detours, tighter time budgets, a lower hack threshold, and an explicit timeout penalty in the true score.
- Why: the first-pass version was too weak because its proxy rewards were often compatible with timely task completion.
- Claim impact: the third environment now supports a clearer delayed-cost proxy-optimization story.
