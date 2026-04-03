"""Schema registry driven by environment modules."""

from __future__ import annotations

from typing import Dict, Iterable

from src.envs import ENV_REGISTRY


def get_task_schema(env_name: str):
    return ENV_REGISTRY[env_name].schema()


def export_all_schemas(env_names: Iterable[str] | None = None) -> Dict[str, dict]:
    names = list(env_names) if env_names is not None else list(ENV_REGISTRY.keys())
    return {name: get_task_schema(name).to_dict() for name in names}

