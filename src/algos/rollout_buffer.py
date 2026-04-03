"""Single-environment rollout buffer with GAE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    values: np.ndarray


class RolloutBuffer:
    def __init__(self, obs_dim: int, capacity: int, gamma: float, gae_lambda: float):
        self.obs_dim = obs_dim
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        idx = self.ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.ptr += 1

    def compute(self, last_value: float) -> RolloutBatch:
        advantages = np.zeros(self.ptr, dtype=np.float32)
        last_advantage = 0.0
        for t in reversed(range(self.ptr)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * next_nonterminal * last_advantage
            advantages[t] = last_advantage
        returns = advantages + self.values[: self.ptr]
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return RolloutBatch(
            observations=self.observations[: self.ptr].copy(),
            actions=self.actions[: self.ptr].copy(),
            log_probs=self.log_probs[: self.ptr].copy(),
            returns=returns.copy(),
            advantages=advantages.copy(),
            values=self.values[: self.ptr].copy(),
        )
