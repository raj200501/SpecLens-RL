#!/usr/bin/env python3
"""Stage 1 minimal paper matrix."""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import ROOT
from scripts.evaluate import evaluate_run
from scripts.make_paper import build_paper_assets
from scripts.make_plots import build_plots
from scripts.make_tables import build_tables
from scripts.train import train_run
from scripts.train_monitor import run_monitor_pipeline
from src.utils.config import load_config
from src.utils.io_utils import read_json, write_json
from src.utils.resource_guard import check_resource_guard


PROFILE_TO_CONFIG = {
    "PROFILE_FULL_SAFE": "profile_full_safe.yaml",
    "PROFILE_LEAN_SAFE": "profile_lean_safe.yaml",
    "PROFILE_ULTRA_LEAN": "profile_ultra_lean.yaml",
    "PROFILE_SCAFFOLD_ONLY": None,
}


def main() -> None:
    preflight = read_json(ROOT / "results" / "preflight.json")
    profile = preflight["selected_profile"]
    if profile == "PROFILE_SCAFFOLD_ONLY":
        summary = {"stage": "stage1", "status": "skipped", "reason": "profile is scaffold-only"}
        write_json(ROOT / "results" / "stage1_summary.json", summary)
        print(json.dumps(summary))
        return
    profile_cfg = load_config(PROFILE_TO_CONFIG[profile])
    guard = check_resource_guard(ROOT, ROOT / "results", profile)
    if not guard.allowed:
        raise RuntimeError(guard.reason)
    for env_name in profile_cfg["environments"]:
        for method_name in profile_cfg["methods"]:
            for seed in range(int(profile_cfg["seed_count"])):
                train_run(env_name, method_name, seed, int(profile_cfg["stage1_train_steps"]), profile)
                evaluate_run(env_name, method_name, seed, int(profile_cfg["eval_episodes"]))
        run_monitor_pipeline(env_name)
    build_tables()
    build_plots()
    build_paper_assets()
    summary = {
        "stage": "stage1",
        "status": "completed",
        "profile": profile,
        "environments": profile_cfg["environments"],
        "methods": profile_cfg["methods"],
        "seed_count": profile_cfg["seed_count"],
        "train_steps": profile_cfg["stage1_train_steps"],
    }
    write_json(ROOT / "results" / "stage1_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
