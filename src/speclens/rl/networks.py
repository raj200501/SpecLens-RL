"""Actor-critic network shared by every SpecLens-RL agent."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Convolutional actor-critic for grid observations.

    Observations are ``n_planes`` one-hot ``size x size`` planes followed by a
    few scalar features (see :class:`speclens.envs.GridTask`). Two 3x3
    convolutions read the planes; their output is concatenated with the
    scalars and passed through one hidden layer to the policy and value heads.
    """

    def __init__(self, n_planes: int, size: int, n_scalars: int, n_actions: int, hidden: int = 256, channels: int = 16):
        super().__init__()
        self.n_planes, self.size, self.n_scalars = n_planes, size, n_scalars
        self.n_actions, self.hidden, self.channels = n_actions, hidden, channels
        self.conv = nn.Sequential(
            nn.Conv2d(n_planes, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(nn.Linear(channels * size * size + n_scalars, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    @classmethod
    def for_env(cls, env, **kwargs) -> ActorCritic:
        task = env.unwrapped
        return cls(len(task.planes), task.size, len(task.scalar_features), task.action_space.n, **kwargs)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        n_grid = self.n_planes * self.size * self.size
        grid = obs[:, :n_grid].reshape(-1, self.n_planes, self.size, self.size)
        h = self.trunk(torch.cat([self.conv(grid), obs[:, n_grid:]], dim=1))
        return Categorical(logits=self.policy_head(h)), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, generator: torch.Generator | None = None) -> np.ndarray:
        """Select actions for a batch of observations (numpy in, numpy out)."""
        dist, _ = self(torch.as_tensor(obs, dtype=torch.float32))
        if deterministic:
            return dist.probs.argmax(dim=-1).numpy()
        return torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1).numpy()

    def config(self) -> dict[str, int]:
        return {
            "n_planes": self.n_planes,
            "size": self.size,
            "n_scalars": self.n_scalars,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "channels": self.channels,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> ActorCritic:
        model = cls(**payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
