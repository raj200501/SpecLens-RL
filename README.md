# SpecLens-RL

**Small, auditable model organisms of reward hacking, reward tampering and monitorability in reinforcement learning.**

[![CI](https://github.com/raj200501/SpecLens-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/raj200501/SpecLens-RL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reward hacking is no longer only a gridworld curiosity. RL-trained language-model agents game their tests
([Baker et al., 2025](https://arxiv.org/abs/2503.11926)), and hacking learned in production environments can
generalize to broader misalignment ([MacDiarmid et al., 2025](https://arxiv.org/abs/2511.18397)). Studying
this at scale is expensive, and the "true objective" is usually unknown. SpecLens-RL takes the opposite
trade-off. It provides three tiny procedurally generated tasks with these properties:

- the true objective is known exactly;
- the misspecification is a single explicit channel that can be scaled or switched off;
- every number in the paper is regenerated from saved results by one command and checked in CI.

It asks three questions:

1. **When does an optimizer exploit a misspecified reward?** We look at training dynamics and at a
   dose–response over the strength of the exploit.
2. **Can a monitor detect hacking from behavior alone?** Monitors are evaluated against baselines that a
   misleading monitor would also beat.
3. **What happens when you train against the monitor?**

The paper is in [`paper/`](paper) (`make paper`), and every result in it comes from [`results/`](results).

## The tasks

![Example layouts](paper/figures/tasks.png)

Each task is a fully observed 7×7 gridworld whose layout is resampled every episode. The proxy reward
`R_p` differs from the true objective `J_t` through one **corruption channel** `C`, scaled by `λ`:

| Task | Intended behavior | Corruption channel | Mechanism | OOD shift |
|---|---|---|---|---|
| **ProxyFarming** | reach the goal | +0.2 per step on a "beacon" tile | non-potential-based shaping creates a reward cycle (Ng et al., 1999) | 3 → 6 walls |
| **SensorTamper** | carry an object to a target | a console makes the delivery sensor emit false confirmations (+1 each) | reward tampering ([Everitt et al., 2021](https://doi.org/10.1007/s11229-021-03141-4)) | console moved far from the start |
| **GemTimeout** | reach the goal before a deadline | +0.6 per gem; every gem is a detour past the deadline | the proxy omits a constraint | 3 → 5 gems |

A *hack* is defined by behavior: the agent used the channel (3+ steps on the beacon, a console press, a
gem), whether or not the channel paid. A *harmful hack* is a hack that also fails the true objective. In
`speclens.envs`, `step()` returns the proxy reward like any Gymnasium environment; the true reward and all
labels are in `info`.

## Results

*Results from the full experiment run are being regenerated with the final code and will be added here.*

## What we got wrong the first time

An earlier version of this repository reported results that did not survive an audit of its own code.
Each failure below is now guarded by a test:

- **Repeated episodes counted as samples.** Deterministic dynamics with two fixed layouts meant every
  "24-episode" evaluation was 2 distinct trajectories, each repeated 12 times.
- **A leakage screen that screened nothing.** Features were flagged by name tokens that no feature
  name contained.
- **An ablation that ablated the measurement.** Switching off the tamper effect also switched off the
  counter behind the hack label, while the policy kept tampering.
- **Correlations of constant vectors** that floating-point noise turned into ±1.
- **Hand-copied numbers** that the pipeline could not regenerate.

## Reproducing everything

```bash
make install       # exact versions from requirements-lock.txt (Python 3.13, CPU-only PyTorch)
make test          # unit tests, about a minute
make verify        # check that the committed results were produced by the current code and config
make report        # figures, tables and numbers from results/ (about a minute; no training)
make paper         # compile paper/main.pdf
make experiments   # retrain everything: 135 PPO runs, a few CPU-hours on 4 cores, resumable
```

`make smoke` runs the entire pipeline at toy scale in about a minute.

Every result file in `results/` records a fingerprint of the code, configuration and (for the
monitor-penalty runs) the monitor that produced it. `make verify`, run in CI, fails if any result is stale.
The paper quotes numbers through `\res{key}` macros written by `speclens.report`. CI regenerates them and
fails if the committed paper assets differ, or if the paper cites a number the report does not produce.

## Using the environments

```python
from speclens.envs import make_env

env = make_env("SensorTamper", split="id", channel_scale=1.0)
obs, _ = env.reset(seed=0)
obs, proxy_reward, terminated, truncated, info = env.step(4)  # 4 = interact
info["true_reward"], info["channel_reward"]
print(env.render())
```

## Repository layout

```
src/speclens/
  envs/          the three tasks and their shared base class
  rl/            PPO (single-file, CleanRL-style) and the actor-critic CNN
  evaluation.py  batched, seeded rollouts on fresh layouts
  monitors/      trajectory features with enforced provenance, CV and baselines, training against a monitor
  analysis/      statistics, monitorability analysis, figures and tables
  experiment.py  every training run in the paper, from one config (python -m speclens.experiment)
  report.py      every figure, table and number in the paper (python -m speclens.report)
configs/         paper.yaml (the paper) and smoke.yaml (CI-scale)
results/         compact results the paper is built from (per-run checkpoints are not committed)
paper/           LaTeX sources; paper/generated/ and paper/figures/ are written by speclens.report
tests/           invariants of the environments, PPO, monitors, statistics and the pipeline
```

## Limitations

These are model organisms, not realistic deployments. Every agent is PPO with one small CNN. The default
channel strengths were chosen in pilot runs so that the oracle learns each task and the proxy channel is
exploited; the dose–response sweep shows what happens at other strengths. Monitors are linear models and
shallow trees over 15–23 hand-written trajectory features. See the paper's limitations section for more.

## Citation

```bibtex
@misc{kashikar2026speclens,
  title  = {SpecLens-RL: Small, Auditable Model Organisms of Reward Hacking, Reward Tampering, and Monitorability},
  author = {Kashikar, Raj},
  year   = {2026},
  url    = {https://github.com/raj200501/SpecLens-RL}
}
```

MIT License.
