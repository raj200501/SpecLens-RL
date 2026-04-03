#!/usr/bin/env python3
"""Repository smoke checks."""

from __future__ import annotations

import json

from _bootstrap import ROOT
from src.envs import ENV_REGISTRY, make_env
from src.schema.registry import export_all_schemas


def main() -> None:
    payload = {"envs": list(ENV_REGISTRY.keys()), "schemas": export_all_schemas()}
    for env_name in ENV_REGISTRY:
        env = make_env(env_name, split="id")
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        obs, reward, terminated, truncated, info = env.step(4)
        assert isinstance(reward, float)
        assert "episode_summary" in info
    print(json.dumps({"status": "ok", "root": str(ROOT), "payload_keys": list(payload.keys())}))


if __name__ == "__main__":
    main()
