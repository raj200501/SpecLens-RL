"""System inspection utilities used by preflight and guards."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def optional_import_version(module_name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return {"available": True, "version": str(version)}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def _battery_info() -> Dict[str, Any]:
    if sys.platform != "darwin":
        return {"detected": False}
    try:
        completed = subprocess.run(
            ["pmset", "-g", "batt"],
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout.strip()
        low = output.lower()
        if "battery power" in low:
            return {"detected": True, "on_battery": True, "raw": output}
        if "ac power" in low:
            return {"detected": True, "on_battery": False, "raw": output}
        return {"detected": False, "raw": output}
    except Exception as exc:
        return {"detected": False, "error": repr(exc)}


def _total_ram_bytes() -> int | None:
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            return int(completed.stdout.strip())
        except Exception:
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size
    except Exception:
        return None


def get_repo_size_bytes(root: str | Path = ROOT) -> int:
    total = 0
    root_path = Path(root)
    for path in root_path.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def get_free_disk_bytes(path: str | Path = ROOT) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def profile_from_preflight(preflight: Dict[str, Any]) -> str:
    free_gb = preflight["system"]["free_disk_bytes"] / (1024 ** 3)
    has_torch = preflight["packages"]["torch"]["available"]
    has_gym = preflight["packages"]["gymnasium"]["available"]
    has_minigrid = preflight["packages"]["minigrid"]["available"]
    logical_cores = preflight["system"]["logical_cpu_cores"] or 0
    if free_gb < 2 or not (has_torch and has_gym and has_minigrid):
        return "PROFILE_SCAFFOLD_ONLY"
    if free_gb < 4:
        return "PROFILE_ULTRA_LEAN"
    if free_gb < 8 or logical_cores < 4:
        return "PROFILE_LEAN_SAFE"
    return "PROFILE_FULL_SAFE"


def collect_system_info(root: str | Path = ROOT) -> Dict[str, Any]:
    root_path = Path(root)
    return {
        "os": platform.platform(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "apple_silicon": platform.system() == "Darwin" and platform.machine() in {"arm64", "arm"},
        "total_ram_bytes": _total_ram_bytes(),
        "logical_cpu_cores": os.cpu_count(),
        "free_disk_bytes": get_free_disk_bytes(root_path),
        "repo_size_bytes": get_repo_size_bytes(root_path),
        "cwd": str(root_path),
        "battery": _battery_info(),
    }


def collect_preflight(root: str | Path = ROOT) -> Dict[str, Any]:
    system = collect_system_info(root)
    packages = {
        "torch": optional_import_version("torch"),
        "gymnasium": optional_import_version("gymnasium"),
        "minigrid": optional_import_version("minigrid"),
    }
    preflight = {"system": system, "packages": packages}
    preflight["selected_profile"] = profile_from_preflight(preflight)
    return preflight


def render_preflight_markdown(preflight: Dict[str, Any]) -> str:
    system = preflight["system"]
    packages = preflight["packages"]
    lines = [
        "# Preflight Audit",
        "",
        f"- Selected profile: `{preflight['selected_profile']}`",
        f"- OS: `{system['os']}`",
        f"- Python: `{system['python_version']}`",
        f"- Apple Silicon: `{system['apple_silicon']}`",
        f"- Total RAM bytes: `{system['total_ram_bytes']}`",
        f"- Logical CPU cores: `{system['logical_cpu_cores']}`",
        f"- Free disk bytes: `{system['free_disk_bytes']}`",
        f"- Repo size bytes: `{system['repo_size_bytes']}`",
        f"- Battery detection: `{json.dumps(system['battery'])}`",
        f"- torch import: `{packages['torch']}`",
        f"- gymnasium import: `{packages['gymnasium']}`",
        f"- minigrid import: `{packages['minigrid']}`",
    ]
    return "\n".join(lines) + "\n"
