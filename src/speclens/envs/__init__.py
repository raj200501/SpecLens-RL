"""SpecLens-RL environments."""

from __future__ import annotations

from speclens.envs.base import ACTION_NAMES, SPLITS, EpisodeStats, GridTask, Layout
from speclens.envs.gem_timeout import GemTimeout
from speclens.envs.proxy_farming import ProxyFarming
from speclens.envs.sensor_tamper import SensorTamper

ENVS: dict[str, type[GridTask]] = {cls.name: cls for cls in (ProxyFarming, SensorTamper, GemTimeout)}


def make_env(name: str, **kwargs) -> GridTask:
    """Instantiate a task by name, e.g. ``make_env("SensorTamper", split="ood")``."""
    try:
        return ENVS[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown environment {name!r}; choose from {sorted(ENVS)}") from None


__all__ = [
    "ACTION_NAMES",
    "ENVS",
    "SPLITS",
    "EpisodeStats",
    "GemTimeout",
    "GridTask",
    "Layout",
    "ProxyFarming",
    "SensorTamper",
    "make_env",
]
