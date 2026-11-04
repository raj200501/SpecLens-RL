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
- the misspecification is a single explicit channel that can be scaled or switched off (GemTimeout's proxy
  also omits a deadline, which matters only once the channel pays);
- every experimental result cited in the paper is regenerated from saved results by one command and
  checked in CI.

It asks three questions — when does an optimizer exploit a misspecified reward, can a monitor detect
hacking from behavior alone, and what happens when you train against the monitor — and finds:

1. **Hacking emerges in three different ways.** ProxyFarming agents first master the intended task and
   abandon it hundreds of thousands of steps later; SensorTamper agents learn to tamper instead of the
   task; GemTimeout agents chase gems almost from the start. Proxy return never drops when hacking takes
   over, so the training signal gives no warning. The pattern depends on the channel's strength, not just
   the mechanism: at twice the default, ProxyFarming looks like SensorTamper.
2. **What the reward pays for does not predict what PPO learns.** Solving sampled layouts exactly shows
   that the proxy-optimal policy already hacks at a small fraction of the default channel strength, while
   within its training budget PPO starts hacking only at several times that in two of the tasks. Where PPO hacks in ProxyFarming and
   SensorTamper, it gives up more of the true objective than the proxy optimum would.
3. **Pooled monitor AUCs can hide weak monitoring within a policy.** Monitors that see only positions and actions reach
   near-perfect held-out AUCs, but when honest reference policies are pooled in, a baseline that knows only
   *what kind of policy* produced an episode (Proxy or Oracle PPO, and at which training stage) comes within
   a few hundredths of AUC. Within the episodes of a single
   policy, the monitors hold up in two tasks but are much weaker in ProxyFarming, where hacking emerges late.
4. **Training against the monitor removes hacking in one task and hides it in another.** In GemTimeout,
   penalized agents keep collecting gems but pad their episodes with a no-op action whose frequency barely
   varied in the monitor's training data, and the monitor stops catching their hacks. The evasion is specific
   to that monitor: one the agents never trained against still catches most of the hacks, and a simple
   check for inputs outside the monitor's training range flags them without any labels.

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf) (sources in [`paper/`](paper)). Every experimental result in
it is regenerated from [`results/`](results) and the configuration by one command.

## The tasks

![Example layouts](paper/figures/tasks.png)

Each task is a fully observed 7×7 gridworld whose layout is resampled every episode. The proxy reward
`R_p` differs from the true objective `J_t` through one **corruption channel** `C`, scaled by `λ` (and, in
GemTimeout, by the deadline the proxy omits):

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

| | |
|---|---|
| ![Training dynamics](paper/figures/training_curves.png) | ![Dose–response](paper/figures/dose_response.png) |
| *Training dynamics over 10 seeds (thin lines: individual Proxy PPO runs).* | *Hack and success rates as the corruption channel is scaled, next to the exact proxy-optimal policy.* |
| ![Monitors](paper/figures/monitors.png) | ![Training against the monitor](paper/figures/monitor_penalty.png) |
| *Held-out monitor AUC against a policy-identity baseline, the best single feature, and the AUC within one Proxy PPO snapshot.* | *Proxy PPO penalized by the behavior monitor: true hack rate vs. what the monitor flags.* |

<!-- results:begin -->
*Generated by `python -m speclens.report` from `results/`; CI fails if it is out of date.*

**Final policies** on fresh in-distribution layouts (mean over training runs, 95% *t*-interval):

| Task | Policy | Success | Hack rate | True return |
|---|---|---|---|---|
| ProxyFarming | Proxy PPO | 11% (2–20) | 85% (76–95) | -0.26 |
|  | Oracle PPO | 99% (98–99) | 0% (0–0) | 0.91 |
| SensorTamper | Proxy PPO | 0% (0–0) | 99% (98–99) | -0.40 |
|  | Oracle PPO | 98% (98–99) | 0% (0–0) | 0.86 |
| GemTimeout | Proxy PPO | 6% (5–7) | 92% (91–94) | -0.12 |
|  | Oracle PPO | 95% (93–97) | 0% (0–0) | 0.88 |

**Dose–response**: hack rate of Proxy PPO / of the exact proxy-optimal policy as the channel is scaled by λ (Proxy PPO: 5 runs per λ; optimum: 100 layouts):

| Task | λ = 0 | λ = 0.25 | λ = 0.5 | λ = 1 | λ = 2 |
|---|---|---|---|---|---|
| ProxyFarming | 0% / 0% | 0% / 100% | 1% / 100% | 89% / 100% | 88% / 100% |
| SensorTamper | 0% / 0% | 99% / 100% | 99% / 100% | 99% / 100% | 90% / 100% |
| GemTimeout | 0% / 0% | 0% / 100% | 56% / 100% | 93% / 100% | 96% / 100% |

**Hack monitors** that see only positions and actions (logistic regression, held-out seeds):

| Task | Monitor AUC | Policy-identity baseline | Best single feature | AUC within one Proxy PPO snapshot |
|---|---|---|---|---|
| ProxyFarming | 0.981 | 0.948 | 0.965 | 0.890 |
| SensorTamper | 0.998 | 0.990 | 0.998 | 0.973 |
| GemTimeout | 0.993 | 0.944 | 0.962 | 0.986 |

**Training against the monitor**: Proxy PPO penalized by the behavior monitor's hack probability:

| Task | Hack rate | Hacks the monitor misses | Monitor recall on hacks: unpenalized → penalized |
|---|---|---|---|
| ProxyFarming | 1% (0–1) | 1% of episodes | 97% → no run hacks enough to measure |
| SensorTamper | 19% (0–71) | 13% of episodes | 100% → 28% (1 of 5 runs hack) |
| GemTimeout | 65% (56–75) | 64% of episodes | 97% → 1% |

Compute: 135 PPO runs, 16 CPU-hours.
<!-- results:end -->

## What we got wrong the first time

An earlier version of this repository reported results that did not survive an audit of its own code.
Each failure below is now guarded by a test or a CI check, or the offending metric was removed:

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
make verify        # check that every committed run's fingerprint matches the current code and config
make report        # figures, tables, numbers and the README results from results/ (a minute or two)
make paper         # compile paper/main.pdf
make experiments   # retrain everything (see the compute line under Results); resumable
```

`make smoke` runs the entire pipeline at toy scale in about a minute.

Every training run in `results/runs.csv` and `results/penalty_runs.csv` records a fingerprint of the code,
configuration and (for the monitor-penalty runs) the monitor that produced it. `make verify`, run in CI,
fails if any run is stale.
The paper quotes numbers through `\res{key}` macros written by `speclens.report`, which also writes the
results tables in this README. CI regenerates both and fails if the committed versions differ, or if the
paper cites a number the report does not produce. Training is deterministic given a seed on the same CPU
instruction set and library versions: retraining all 15 monitor-penalty runs in a fresh container reproduced
every evaluation episode exactly. On other hardware, expect the same results in distribution rather than bit
for bit.

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
  analysis/      statistics, monitorability, the exact proxy-optimal policy, monitor evasion, figures, tables
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
shallow trees over 15–23 hand-written trajectory features, and the monitor-penalty experiment uses one
penalty weight, a fixed linear monitor and five seeds per task. See the paper's limitations section.

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
