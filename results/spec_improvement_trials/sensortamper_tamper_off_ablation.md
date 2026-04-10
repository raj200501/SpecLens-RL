# SensorTamper Tamper-Off Ablation

## What changed

- Added a minimal evaluation-only flag to `SensorTamper-v0`:
  - `tamper_effect_enabled=False`
- This disables the corruption-channel effect during evaluation without
  retraining policies.
- Added a focused environment test covering the disabled-effect behavior.

## What was run

- `./.venv/bin/pytest -q tests/test_envs.py`
- Deterministic evaluation from existing saved checkpoints for:
  - `PROXY_PPO`
  - `ORACLE_TRUE_PPO`
- Environment setting:
  - `SensorTamper-v0`
  - `tamper_effect_enabled=False`
  - `24` ID and `24` OOD episodes per seed
  - seeds `0,1,2`

## What evidence was generated

- Trial artifact root:
  - `results/spec_improvement_trials/sensortamper_tamper_off/`
- Comparison summary:
  - `results/spec_improvement_trials/sensortamper_tamper_off/comparison_summary.csv`

Key result:
- Baseline `PROXY_PPO` on SensorTamper ID:
  - `true_score = -3.346`
  - `hack_rate = 0.833`
  - `tamper_rate = 0.833`
  - `proxy_return = 15.9`
- Tamper-off `PROXY_PPO` on SensorTamper ID:
  - `true_score = 0.288`
  - `hack_rate = 0.0`
  - `tamper_rate = 0.0`
  - `proxy_return = -0.45`
- `ORACLE_TRUE_PPO` is essentially unchanged by the ablation:
  - ID `true_score = 1.0` in both settings

## Whether it helped

Yes.

This is a real strengthening pass because it turns the corruption-channel claim
from largely behavioral correlation into a cheap causal intervention: the proxy
exploit collapses when the tamper effect is removed, while the oracle policy is
unchanged.

## Whether it should be included in the paper

Yes.

It deserves a short main-text sentence in the SensorTamper results paragraph.
It sharpens the paper's strongest environment-specific empirical claim without
adding a new method or broad experiment campaign.

## Whether it changes the core story

It strengthens the existing core story.

The paper remains a benchmark and evaluation-methodology paper, but the
SensorTamper corruption result is now more causally persuasive.
