"""Compact PPO trainer for symbolic environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch
from torch import nn

from src.models import ActorCritic

from .losses import ppo_loss
from .rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    reward_mode: str = "proxy"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    rollout_steps: int = 128
    minibatch_size: int = 128
    epochs: int = 4
    hidden_sizes: tuple[int, ...] = (128, 128)
    loop_penalty: float = 0.05
    tamper_penalty: float = 0.2

    @classmethod
    def from_dict(cls, payload: dict):
        fields = cls.__dataclass_fields__.keys()
        filtered = {key: value for key, value in payload.items() if key in fields}
        return cls(**filtered)


class PPOTrainer:
    def __init__(self, env_fn: Callable[[], object], config: PPOConfig, device: str = "cpu"):
        self.env_fn = env_fn
        self.env = env_fn()
        self.config = config
        self.device = torch.device(device)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(self.env.action_space.n)
        self.model = ActorCritic(obs_dim, action_dim, hidden_sizes=config.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _select_training_reward(self, proxy_reward: float, info: Dict[str, object]) -> float:
        summary = info["episode_summary"]
        if self.config.reward_mode == "proxy":
            return float(proxy_reward)
        if self.config.reward_mode == "true":
            return float(info.get("true_step_reward", 0.0))
        if self.config.reward_mode == "conservative_proxy":
            penalty = 0.0
            if summary.get("tamper_used", False):
                penalty += self.config.tamper_penalty
            if summary.get("beacon_interaction_count", 0) > 0 or summary.get("gems_collected", 0) > 2:
                penalty += self.config.loop_penalty
            return float(proxy_reward) - penalty
        raise ValueError(f"Unknown reward mode: {self.config.reward_mode}")

    def train(self, total_steps: int) -> Dict[str, object]:
        obs, _ = self.env.reset(seed=0)
        buffer = RolloutBuffer(
            obs_dim=int(np.prod(self.env.observation_space.shape)),
            capacity=self.config.rollout_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        episode_summaries: List[dict] = []
        updates = max(1, total_steps // self.config.rollout_steps)
        for _ in range(updates):
            buffer.reset()
            for _ in range(self.config.rollout_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.model.act(obs_tensor)
                next_obs, proxy_reward, terminated, truncated, info = self.env.step(int(action.item()))
                train_reward = self._select_training_reward(proxy_reward, info)
                done = terminated or truncated
                buffer.add(obs, int(action.item()), train_reward, done, float(log_prob.item()), float(value.item()))
                obs = next_obs
                if done:
                    episode_summaries.append(info["episode_summary"])
                    obs, _ = self.env.reset(seed=len(episode_summaries))
            with torch.no_grad():
                last_value = self.model.dist_and_value(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))[1].item()
            batch = buffer.compute(last_value=last_value)
            self._update(batch)
        return {
            "total_steps": total_steps,
            "episodes": len(episode_summaries),
            "final_episode": episode_summaries[-1] if episode_summaries else {},
            "episode_summaries": episode_summaries[-25:],
        }

    def _update(self, batch) -> None:
        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        size = obs.shape[0]
        for _ in range(self.config.epochs):
            indices = np.arange(size)
            np.random.shuffle(indices)
            for start in range(0, size, self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                dist, values = self.model.dist_and_value(obs[batch_idx])
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy()
                total, _, _, _ = ppo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs[batch_idx],
                    advantages=advantages[batch_idx],
                    values=values,
                    returns=returns[batch_idx],
                    clip_eps=self.config.clip_eps,
                    value_coef=self.config.value_coef,
                    entropy=entropy,
                    entropy_coef=self.config.entropy_coef,
                )
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.model.act(obs_tensor, deterministic=deterministic)
        return int(action.item())
