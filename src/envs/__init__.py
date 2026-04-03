"""Environment registry."""

from __future__ import annotations

from typing import Dict, Type

from .gem_timeout import GemTimeoutEnv
from .proxy_farming import ProxyFarmingEnv
from .sensor_tamper import SensorTamperEnv


ENV_REGISTRY: Dict[str, Type] = {
    "ProxyFarming-v0": ProxyFarmingEnv,
    "SensorTamper-v0": SensorTamperEnv,
    "GemTimeout-v0": GemTimeoutEnv,
}


def make_env(name: str, **kwargs):
    return ENV_REGISTRY[name](**kwargs)

