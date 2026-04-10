# SensorTamper Conservative Proxy Trial

## What changed

- Ran an isolated `CONSERVATIVE_PROXY_PPO` trial on `SensorTamper-v0` under the
  existing `200000`-step budget.
- Kept the run separate from the main benchmark under:
  - `results/spec_improvement_trials/conservative_proxy_sensor/`

## What was run

- `./.venv/bin/python scripts/train.py --env SensorTamper-v0 --method CONSERVATIVE_PROXY_PPO --seed 0 --steps 200000 --profile PROFILE_FULL_SAFE --results-root results/spec_improvement_trials/conservative_proxy_sensor`
- `./.venv/bin/python scripts/evaluate.py --env SensorTamper-v0 --method CONSERVATIVE_PROXY_PPO --seed 0 --eval-episodes 24 --results-root results/spec_improvement_trials/conservative_proxy_sensor --checkpoint-path results/spec_improvement_trials/conservative_proxy_sensor/runs/SensorTamper-v0/CONSERVATIVE_PROXY_PPO/seed_0/best.ckpt --max-saved-per-split 0`

## What evidence was generated

- Training summary:
  - `results/spec_improvement_trials/conservative_proxy_sensor/runs/SensorTamper-v0/CONSERVATIVE_PROXY_PPO/seed_0/summary.json`
- Deterministic evaluation summary:
  - `results/spec_improvement_trials/conservative_proxy_sensor/runs/SensorTamper-v0/CONSERVATIVE_PROXY_PPO/seed_0/eval_summary.json`
- Episode metrics:
  - `results/spec_improvement_trials/conservative_proxy_sensor/runs/SensorTamper-v0/CONSERVATIVE_PROXY_PPO/seed_0/episodes.csv.gz`
- Baseline comparison source:
  - `results/second_pass_aggregated_main_results.csv`

Observed first-seed outcome:
- Trial `ID true score`: about `-4.088`
- Trial `ID hack incidence`: `1.000`
- Trial `ID tamper utilization`: `1.000`
- Saved Proxy PPO baseline `ID true score`: `-3.346`
- Saved Proxy PPO baseline `ID hack incidence`: `0.833`

## Whether it helped

No.

The first seed moved in the wrong direction relative to the saved proxy
baseline: lower true score and higher hack/tamper incidence. That is enough to
reject the ablation for paper use without spending more compute on the
remaining seeds.

## Whether it should be included in the paper

No.

Including this would dilute the diagnosis-first story and add a weak method
comparison that does not improve the benchmark claim.

## Whether it changes the core story

No.

This is a dead end. The paper remains strongest without it.
