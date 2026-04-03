# Override Log

## 2026-04-03

- Override: removed reward-history aggregate features from the monitor channel specs.
- Why: Stage 0 leakage diagnostics on `ProxyFarming-v0` showed that features such as `reward_mean` trivially predicted the hack label, making monitorability results scientifically weak.
- Claim impact: monitorability claims are now narrower. They characterize detection from behavior and compact progress summaries rather than from direct reward-history aggregates.

- Override: removed benchmark split indicators and risky-zone summary counts from the monitor channels.
- Why: after the first monitor redesign, `split_is_ood` and risky-zone summaries still trivialized hack prediction in Stage 0.
- Claim impact: monitor results now focus on trajectory structure and end-state summaries rather than explicit benchmark metadata or near-label source variables.
