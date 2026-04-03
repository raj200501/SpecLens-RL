"""Common helpers for symbolic grid environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium import spaces

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Position = Tuple[int, int]
ACTIONS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
    4: (0, 0),    # interact / wait
}


@dataclass(frozen=True)
class LayoutVariant:
    name: str
    split: str
    params: Dict[str, object]


def clamp_position(pos: Position, grid_size: int) -> Position:
    return max(0, min(grid_size - 1, pos[0])), max(0, min(grid_size - 1, pos[1]))


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class BaseSymbolicEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}
    action_names = ["up", "down", "left", "right", "interact"]
    obs_channels = ["agent", "goal", "beacon", "tamper", "object", "gem", "obstacle"]

    def __init__(
        self,
        name: str,
        grid_size: int,
        max_steps: int,
        split: str = "id",
        variant_index: int | None = None,
        extra_feature_dim: int = 4,
    ):
        super().__init__()
        self.name = name
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.split = split
        self.variant_index = variant_index
        self.extra_feature_dim = extra_feature_dim
        self.action_space = spaces.Discrete(len(self.action_names))
        obs_dim = grid_size * grid_size * len(self.obs_channels) + extra_feature_dim
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.np_random = np.random.default_rng(0)
        self.current_variant: LayoutVariant | None = None
        self.agent_pos: Position = (0, 0)
        self.goal_pos: Position = (grid_size - 1, grid_size - 1)
        self.obstacles: List[Position] = []
        self.steps = 0
        self.episode_proxy_return = 0.0
        self.last_proxy_reward = 0.0
        self.trajectory: List[dict] = []
        self.summary: Dict[str, object] = {}

    @classmethod
    def schema(cls):
        raise NotImplementedError

    @classmethod
    def variants(cls) -> Dict[str, Sequence[LayoutVariant]]:
        raise NotImplementedError

    def _choose_variant(self, seed: int | None) -> LayoutVariant:
        variants = list(self.variants()[self.split])
        if self.variant_index is not None:
            return variants[self.variant_index % len(variants)]
        if seed is None:
            return variants[0]
        return variants[int(seed) % len(variants)]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.current_variant = self._choose_variant(seed)
        self._reset_from_variant(self.current_variant)
        obs = self._build_observation()
        info = {"env_name": self.name, "split": self.split, "variant_name": self.current_variant.name, "episode_summary": self.get_episode_summary()}
        return obs, info

    def _reset_from_variant(self, variant: LayoutVariant) -> None:
        raise NotImplementedError

    def _step_logic(self, action: int) -> Tuple[float, bool]:
        raise NotImplementedError

    def _build_layers(self) -> np.ndarray:
        layers = np.zeros((len(self.obs_channels), self.grid_size, self.grid_size), dtype=np.float32)
        for pos in self.obstacles:
            layers[self.obs_channels.index("obstacle"), pos[0], pos[1]] = 1.0
        layers[self.obs_channels.index("agent"), self.agent_pos[0], self.agent_pos[1]] = 1.0
        layers[self.obs_channels.index("goal"), self.goal_pos[0], self.goal_pos[1]] = 1.0
        return layers

    def _extra_obs_features(self) -> np.ndarray:
        return np.array(
            [
                min(1.0, self.steps / max(1, self.max_steps)),
                min(1.0, max(0.0, self.episode_proxy_return / max(1, self.max_steps))),
                min(1.0, max(0.0, self.last_proxy_reward + 1.0) / 2.0),
                1.0 if self.split == "ood" else 0.0,
            ],
            dtype=np.float32,
        )

    def _build_observation(self) -> np.ndarray:
        layers = self._build_layers().reshape(-1)
        return np.concatenate([layers, self._extra_obs_features()], axis=0).astype(np.float32)

    def _move_agent(self, action: int) -> None:
        delta = ACTIONS[action]
        candidate = clamp_position((self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]), self.grid_size)
        if candidate not in self.obstacles:
            self.agent_pos = candidate

    def step(self, action: int):
        self.steps += 1
        self._move_agent(action) if action != 4 else None
        previous_true_score = float(self.summary.get("true_score", 0.0))
        proxy_reward, terminated = self._step_logic(action)
        self.last_proxy_reward = float(proxy_reward)
        self.episode_proxy_return += float(proxy_reward)
        truncated = self.steps >= self.max_steps and not terminated
        obs = self._build_observation()
        summary = self.get_episode_summary()
        true_step_reward = float(summary.get("true_score", 0.0) - previous_true_score)
        self.trajectory.append(
            {
                "step": self.steps,
                "action": int(action),
                "agent_pos": list(self.agent_pos),
                "proxy_reward": float(proxy_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "summary_snapshot": summary,
            }
        )
        info = {
            "env_name": self.name,
            "split": self.split,
            "variant_name": self.current_variant.name if self.current_variant else "unknown",
            "episode_summary": summary,
            "true_step_reward": true_step_reward,
        }
        return obs, float(proxy_reward), bool(terminated), bool(truncated), info

    def get_episode_summary(self) -> Dict[str, object]:
        return dict(self.summary)

    def monitor_context(self) -> Dict[str, object]:
        return {
            "goal_pos": list(self.goal_pos),
            "risky_positions": [],
            "split": self.split,
            "variant_name": self.current_variant.name if self.current_variant else "unknown",
        }

    def render_text(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x, y in self.obstacles:
            grid[x][y] = "#"
        grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
        self._render_entities(grid)
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
        return "\n".join("".join(row) for row in grid)

    def _render_entities(self, grid: List[List[str]]) -> None:
        return

    def render_matplotlib(self, path: str | Path | None = None):
        char_grid = self.render_text().splitlines()
        data = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        color_map = {
            ".": (0.96, 0.96, 0.96),
            "#": (0.20, 0.20, 0.20),
            "G": (0.20, 0.60, 0.20),
            "A": (0.15, 0.25, 0.75),
            "B": (0.95, 0.70, 0.10),
            "T": (0.75, 0.20, 0.20),
            "O": (0.55, 0.35, 0.15),
            "*": (0.75, 0.10, 0.75),
        }
        for i, row in enumerate(char_grid):
            for j, token in enumerate(row):
                data[i, j] = color_map.get(token, (1.0, 1.0, 1.0))
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(data, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{self.name} [{self.split}]")
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        return fig
