"""Runtime resource guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .budget import artifact_budget_bytes, directory_size_bytes
from .system import get_free_disk_bytes


@dataclass
class GuardStatus:
    allowed: bool
    reason: str
    free_disk_bytes: int
    results_size_bytes: int
    artifact_budget_bytes: int


def check_resource_guard(
    project_root: str | Path,
    results_dir: str | Path,
    profile_name: str,
    stop_free_disk_bytes: int = 1024 ** 3,
) -> GuardStatus:
    free_disk = get_free_disk_bytes(project_root)
    results_size = directory_size_bytes(results_dir)
    budget_bytes = artifact_budget_bytes(profile_name)
    if free_disk < stop_free_disk_bytes:
        return GuardStatus(
            allowed=False,
            reason="Free disk below hard stop threshold.",
            free_disk_bytes=free_disk,
            results_size_bytes=results_size,
            artifact_budget_bytes=budget_bytes,
        )
    if results_size > budget_bytes:
        return GuardStatus(
            allowed=False,
            reason="Artifact budget exceeded for active profile.",
            free_disk_bytes=free_disk,
            results_size_bytes=results_size,
            artifact_budget_bytes=budget_bytes,
        )
    return GuardStatus(
        allowed=True,
        reason="Resource guard passed.",
        free_disk_bytes=free_disk,
        results_size_bytes=results_size,
        artifact_budget_bytes=budget_bytes,
    )

