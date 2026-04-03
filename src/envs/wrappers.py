"""Optional environment wrappers."""

from __future__ import annotations

import gymnasium as gym


class EpisodeSummaryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_episode_summary = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_episode_summary = info.get("episode_summary")
        return obs, reward, terminated, truncated, info

