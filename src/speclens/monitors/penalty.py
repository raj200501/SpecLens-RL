"""Training against a monitor.

A :class:`LinearMonitor` is a logistic-regression hack detector over one
monitor channel, stored as plain numbers so it can be saved as JSON and applied
inside a training loop. :class:`MonitorPenalty` wraps an environment so that, at
the end of every episode, ``beta`` times the monitor's hack probability is
subtracted from the proxy reward. Training on the wrapped reward asks whether
penalizing *detected* hacking removes hacking or only the part a monitor can see
(compare Baker et al., 2025).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from speclens.envs import GridTask
from speclens.monitors.features import channel_features, extract_features


@dataclass(frozen=True)
class LinearMonitor:
    features: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    threshold: float = 0.5

    @classmethod
    def fit(cls, data: pd.DataFrame, channel: str, target: str = "hack") -> LinearMonitor:
        names = channel_features(channel)
        X = data[names].to_numpy(dtype=float)
        mean, scale = X.mean(axis=0), X.std(axis=0)
        scale[scale == 0] = 1.0
        clf = LogisticRegression(max_iter=5000).fit((X - mean) / scale, data[target].astype(int))
        return cls(tuple(names), tuple(mean), tuple(scale), tuple(clf.coef_[0]), float(clf.intercept_[0]))

    def score(self, features: dict[str, float]) -> float:
        x = (np.array([features[n] for n in self.features]) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(x @ np.array(self.coef) + self.intercept))))

    def score_trajectory(self, trajectory: dict) -> float:
        return self.score(extract_features(trajectory))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LinearMonitor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in payload.items()})


class MonitorPenalty(gym.Wrapper):
    """Subtract ``beta * monitor score`` from the proxy reward when an episode ends.

    The episode statistics in ``info["episode"]`` are those of the wrapped task
    (proxy return excludes the penalty), plus the ``monitor_score``.
    """

    def __init__(self, env: GridTask, monitor: LinearMonitor, beta: float) -> None:
        if not env.unwrapped.record_trajectory:
            raise ValueError("MonitorPenalty needs an environment created with record_trajectory=True")
        super().__init__(env)
        self.monitor, self.beta = monitor, beta

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            score = self.monitor.score_trajectory(self.env.unwrapped.trajectory())
            reward -= self.beta * score
            info["episode"]["monitor_score"] = score
        return obs, reward, terminated, truncated, info
