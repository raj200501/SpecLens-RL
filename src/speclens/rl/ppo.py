"""Proximal Policy Optimization (Schulman et al., 2017) for the SpecLens-RL tasks.

A deliberately plain, single-file implementation in the style of CleanRL:
synchronous environments stepped in lock-step, GAE(lambda), clipped surrogate
objective, linear learning-rate annealing and per-minibatch advantage
normalization. The only SpecLens-specific choice is ``reward``: agents are
trained either on the proxy reward returned by ``env.step`` or on the true
reward reported in ``info``, with everything else held fixed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from speclens.envs import GridTask
from speclens.rl.networks import ActorCritic

RewardSource = Literal["proxy", "true"]


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 16
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 256
    channels: int = 16

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.rollout_steps

    @property
    def n_updates(self) -> int:
        return max(1, self.total_steps // self.batch_size)

    def as_dict(self) -> dict:
        return asdict(self)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation over a ``(T, n_envs)`` rollout.

    ``dones[t]`` marks that the transition at step ``t`` ended its episode, so
    no value is bootstrapped across it. Returns ``(advantages, returns)``.
    """
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[t] = running
    return advantages, advantages + values


@dataclass
class TrainResult:
    model: ActorCritic
    log: list[dict]  # one row per update: training-episode statistics and losses
    checkpoints: list[tuple[int, dict]]  # (env steps, state_dict)
    wall_clock_sec: float


def train_ppo(
    make_env: Callable[[], GridTask],
    reward: RewardSource,
    config: PPOConfig,
    seed: int,
    checkpoint_steps: Sequence[int] = (),
) -> TrainResult:
    """Train a policy with PPO and return it with its training log.

    Each of the ``config.n_envs`` environments draws layouts from its own RNG
    stream derived from ``seed``, so a run is fully determined by ``seed``.
    ``checkpoint_steps`` lists environment-step counts at which to snapshot
    the policy. Each snapshot is taken at the first update boundary at or after
    its count; counts beyond the last update (``total_steps`` need not be a
    multiple of the batch size) are served by the final policy, so exactly one
    snapshot is returned per requested count.
    """
    if reward not in ("proxy", "true"):
        raise ValueError(f"reward must be 'proxy' or 'true', got {reward!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = [make_env() for _ in range(config.n_envs)]
    env_seeds = np.random.SeedSequence(seed).spawn(config.n_envs)
    obs = np.stack([env.reset(seed=int(s.generate_state(1)[0]))[0] for env, s in zip(envs, env_seeds)])
    model = ActorCritic.for_env(envs[0], hidden=config.hidden, channels=config.channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    T, N = config.rollout_steps, config.n_envs
    buf_obs = np.zeros((T, N, obs.shape[1]), dtype=np.float32)
    buf_actions = np.zeros((T, N), dtype=np.int64)
    buf_logp = np.zeros((T, N), dtype=np.float32)
    buf_rewards = np.zeros((T, N), dtype=np.float32)
    buf_dones = np.zeros((T, N), dtype=np.float32)
    buf_values = np.zeros((T, N), dtype=np.float32)

    pending = sorted(checkpoint_steps)
    checkpoints: list[tuple[int, dict]] = []
    log: list[dict] = []
    start = time.time()
    for update in range(config.n_updates):
        if config.anneal_lr:
            optimizer.param_groups[0]["lr"] = config.learning_rate * (1.0 - update / config.n_updates)
        finished: list[dict] = []
        for t in range(T):
            with torch.no_grad():
                dist, value = model(torch.as_tensor(obs))
                action = dist.sample()
            buf_obs[t] = obs
            buf_actions[t] = action.numpy()
            buf_logp[t] = dist.log_prob(action).numpy()
            buf_values[t] = value.numpy()
            for i, env in enumerate(envs):
                next_obs, proxy_r, terminated, truncated, info = env.step(int(buf_actions[t, i]))
                buf_rewards[t, i] = proxy_r if reward == "proxy" else info["true_reward"]
                buf_dones[t, i] = float(terminated or truncated)
                if terminated or truncated:
                    finished.append(info["episode"])
                    next_obs, _ = env.reset()
                obs[i] = next_obs
        with torch.no_grad():
            _, last_value = model(torch.as_tensor(obs))
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_value.numpy(), config.gamma, config.gae_lambda
        )
        losses = _update(model, optimizer, config, rng, buf_obs, buf_actions, buf_logp, advantages, returns)

        steps = (update + 1) * config.batch_size
        row = {"update": update, "env_steps": steps, **losses, "episodes": len(finished)}
        for key in ("proxy_return", "true_return", "success", "hack", "exploit_events", "length"):
            row[key] = float(np.mean([ep[key] for ep in finished])) if finished else float("nan")
        log.append(row)
        while pending and (steps >= pending[0] or update == config.n_updates - 1):
            pending.pop(0)
            checkpoints.append((steps, {k: v.clone() for k, v in model.state_dict().items()}))
    return TrainResult(model=model, log=log, checkpoints=checkpoints, wall_clock_sec=time.time() - start)


def _update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    rng: np.random.Generator,
    obs: np.ndarray,
    actions: np.ndarray,
    old_logp: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
) -> dict[str, float]:
    b_obs = torch.as_tensor(obs.reshape(-1, obs.shape[-1]))
    b_actions = torch.as_tensor(actions.reshape(-1))
    b_logp = torch.as_tensor(old_logp.reshape(-1))
    b_adv = torch.as_tensor(advantages.reshape(-1))
    b_returns = torch.as_tensor(returns.reshape(-1))
    n = b_obs.shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_minibatches = 0
    for _ in range(config.epochs):
        order = rng.permutation(n)
        for begin in range(0, n, config.minibatch_size):
            idx = torch.as_tensor(order[begin : begin + config.minibatch_size])
            dist, values = model(b_obs[idx])
            logp = dist.log_prob(b_actions[idx])
            log_ratio = logp - b_logp[idx]
            ratio = log_ratio.exp()
            adv = b_adv[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - config.clip_eps, 1 + config.clip_eps) * adv).mean()
            value_loss = 0.5 * (b_returns[idx] - values).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > config.clip_eps).float().mean().item()
            n_minibatches += 1
    return {key: value / n_minibatches for key, value in stats.items()}
