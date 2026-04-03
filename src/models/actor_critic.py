"""Compact actor-critic network."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .encoders import MLPEncoder


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, hidden_sizes=hidden_sizes)
        self.policy_head = nn.Linear(self.encoder.output_dim, action_dim)
        self.value_head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def dist_and_value(self, obs: torch.Tensor):
        logits, values = self(obs)
        return Categorical(logits=logits), values

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self.dist_and_value(obs)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

