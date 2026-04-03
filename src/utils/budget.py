"""Artifact budget tracking."""

from __future__ import annotations

from pathlib import Path

from .system import get_free_disk_bytes


PROFILE_BUDGET_MB = {
    "PROFILE_FULL_SAFE": 3072,
    "PROFILE_LEAN_SAFE": 1536,
    "PROFILE_ULTRA_LEAN": 700,
    "PROFILE_SCAFFOLD_ONLY": 250,
}


def directory_size_bytes(path: str | Path) -> int:
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for item in root.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def artifact_budget_bytes(profile_name: str) -> int:
    return PROFILE_BUDGET_MB[profile_name] * 1024 * 1024


def artifact_usage_ok(results_dir: str | Path, profile_name: str) -> bool:
    return directory_size_bytes(results_dir) <= artifact_budget_bytes(profile_name)


def free_disk_ok(path: str | Path, minimum_bytes: int = 1024 ** 3) -> bool:
    return get_free_disk_bytes(path) >= minimum_bytes

