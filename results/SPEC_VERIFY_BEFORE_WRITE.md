# Spec Verify Before Write

Date: `2026-04-10`

This map covers the new or materially revised claims introduced in the NeurIPS
workshop hardening pass. Claims without a concrete repo artifact should not be
added to the paper.

| Claim text | Supporting artifact | Exact file path | Support |
| --- | --- | --- | --- |
| SpecLens-RL is a benchmark and evaluation protocol built around the explicit task tuple `(M, R_p, J_t, C, M_mon, Delta, Pi)`. | Repo scope and task tuple definition. | `README.md` | direct |
| The benchmark is instantiated in three symbolic environments: ProxyFarming-v0, SensorTamper-v0, and GemTimeout-v0. | Repo environment listing and environment summary table source. | `README.md` | direct |
| The paper is not a new RL optimizer paper; the method family is deliberately small and diagnosis-oriented. | Repo method set and benchmark framing. | `configs/default.yaml` | direct |
| Across all three tasks, proxy-trained policies diverge sharply from oracle policies on the hidden objective. | Aggregated second-pass benchmark results. | `results/second_pass_aggregated_main_results.csv` | direct |
| SensorTamper-v0 exposes a behaviorally active corruption channel rather than a dormant tamper feature. | Aggregated tampering metrics show nonzero tamper use and a large negative true-score delta. | `results/second_pass_aggregated_tampering_results.csv` | direct |
| Valid leakage-cleared monitor evidence is narrow because only SensorTamper-v0 yields a nondegenerate hack-label mix under the current proxy-only protocol. | Cross-environment monitor-scope summary generated from proxy-only monitor datasets and monitor outputs. | `results/second_pass/aggregated_monitor_scope_results.csv` | direct |
| Under the proxy-only ID monitor protocol, SensorTamper has `58/72` positive hack labels, ProxyFarming has `72/72`, and GemTimeout has `71/72`. | Cross-environment monitor-scope summary. | `results/second_pass/aggregated_monitor_scope_results.csv` | direct |
| Failure labels collapse to a single class in all three environments under the current proxy-only ID monitor protocol. | Cross-environment monitor-scope summary. | `results/second_pass/aggregated_monitor_scope_results.csv` | direct |
| SensorTamper-v0 restricted logistic regression reaches ROC-AUC `0.936` and PR-AUC `0.962`; the corresponding rich-channel model reaches ROC-AUC `0.964` and PR-AUC `0.986`. | Aggregated second-pass monitor results. | `results/second_pass_aggregated_monitor_results.csv` | direct |
| The rich-channel gain is real but modest, so the restricted channel is already behaviorally informative. | Rich-minus-restricted comparison for accepted SensorTamper monitor rows. | `results/second_pass_aggregated_monitor_results.csv` | inferential |
| ProxyFarming-v0 and GemTimeout-v0 do not support accepted grouped hack-monitor evaluation under the current proxy-only protocol. | Monitor-scope summary and degenerate monitor result rows. | `results/second_pass/aggregated_monitor_scope_results.csv` | direct |
| The benchmark remains laptop-scale and controlled-compute rather than a large experiment campaign. | Second-pass runtime summary and repo framing. | `results/second_pass/second_pass_summary.json` | inferential |

## Excluded claim candidates

- `CONSERVATIVE_PROXY_PPO` improves SensorTamper behavior.
  Excluded because the first isolated trial moved in the wrong direction relative
  to the saved Proxy PPO baseline and was not strong enough to justify paper
  inclusion.
