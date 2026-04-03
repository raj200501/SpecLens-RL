#!/usr/bin/env python3
"""Stage 0 smoke pipeline."""

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
from src.utils.io_utils import read_json, write_json, write_text


def main() -> None:
    preflight_path = ROOT / "results" / "preflight.json"
    if not preflight_path.exists():
        raise FileNotFoundError("Run scripts/preflight.py first.")
    preflight = read_json(preflight_path)
    profile = preflight["selected_profile"]
    default_cfg = load_config("default.yaml")
    stage0_steps = int(default_cfg["run"]["stages"]["stage0_steps"])
    eval_episodes = 6
    env_name = "ProxyFarming-v0"
    method_name = "PROXY_PPO"
    seed = 0
    run_dir = train_run(env_name, method_name, seed, stage0_steps, profile)
    evaluate_run(env_name, method_name, seed, eval_episodes)
    run_monitor_pipeline(env_name)
    build_tables()
    build_plots()
    build_paper_assets()
    summary = {
        "stage": "stage0",
        "profile": profile,
        "run_dir": str(run_dir),
        "env": env_name,
        "method": method_name,
        "seed": seed,
        "steps": stage0_steps,
    }
    write_json(ROOT / "results" / "stage0_summary.json", summary)
    write_text(ROOT / "results" / "stage0_status.txt", "ok\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
