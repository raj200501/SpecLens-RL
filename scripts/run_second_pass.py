#!/usr/bin/env python3
"""Targeted second-pass rerun for SpecLens-RL."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from _bootstrap import ROOT
from evaluate import evaluate_run
from make_paper import build_paper_assets
from make_plots import build_plots
from make_tables import build_tables
from train import train_run
from train_monitor import run_monitor_pipeline
from src.utils.io_utils import read_json, write_json
from src.utils.resource_guard import check_resource_guard


MAIN_RESULTS_ROOT = ROOT / "results" / "second_pass"
MONITOR_RESULTS_ROOT = MAIN_RESULTS_ROOT / "monitor_eval"
TOP_LEVEL_RESULTS = ROOT / "results"
METHODS = ["PROXY_PPO", "ORACLE_TRUE_PPO"]
SEEDS = [0, 1, 2]
MAIN_EVAL_EPISODES = 24
MONITOR_EVAL_EPISODES = 24
GEM_TIMEOUT_STEPS = 60000


def _profile_name() -> str:
    preflight = read_json(TOP_LEVEL_RESULTS / "preflight.json")
    return str(preflight.get("selected_profile", "PROFILE_SCAFFOLD_ONLY"))


def _stage_summary_path() -> Path:
    return MAIN_RESULTS_ROOT / "second_pass_summary.json"


def _check_guard(profile_name: str) -> None:
    guard = check_resource_guard(ROOT, TOP_LEVEL_RESULTS, profile_name)
    if not guard.allowed:
        raise RuntimeError(guard.reason)


def _write_summary(payload: dict) -> None:
    write_json(_stage_summary_path(), payload)


def _evaluate_existing_env(env_name: str, deterministic: bool, results_root: Path, eval_episodes: int, max_saved_per_split: int) -> None:
    for method_name in METHODS:
        for seed in SEEDS:
            checkpoint = TOP_LEVEL_RESULTS / "runs" / env_name / method_name / f"seed_{seed}" / "best.ckpt"
            evaluate_run(
                env_name,
                method_name,
                seed,
                eval_episodes=eval_episodes,
                results_root=results_root,
                checkpoint_path=checkpoint,
                deterministic=deterministic,
                max_saved_per_split=max_saved_per_split,
            )


def _train_and_evaluate_gem_timeout(profile_name: str) -> None:
    for method_name in METHODS:
        for seed in SEEDS:
            _check_guard(profile_name)
            train_run(
                "GemTimeout-v0",
                method_name,
                seed,
                total_steps=GEM_TIMEOUT_STEPS,
                profile_name=profile_name,
                results_root=MAIN_RESULTS_ROOT,
            )
            evaluate_run(
                "GemTimeout-v0",
                method_name,
                seed,
                eval_episodes=MAIN_EVAL_EPISODES,
                results_root=MAIN_RESULTS_ROOT,
                deterministic=True,
                max_saved_per_split=0,
            )


def _evaluate_proxy_monitor_rollouts(env_name: str) -> None:
    for seed in SEEDS:
        checkpoint_root = MAIN_RESULTS_ROOT if env_name == "GemTimeout-v0" else TOP_LEVEL_RESULTS
        checkpoint = checkpoint_root / "runs" / env_name / "PROXY_PPO" / f"seed_{seed}" / "best.ckpt"
        evaluate_run(
            env_name,
            "PROXY_PPO",
            seed,
            eval_episodes=MONITOR_EVAL_EPISODES,
            results_root=MONITOR_RESULTS_ROOT,
            checkpoint_path=checkpoint,
            deterministic=False,
            max_saved_per_split=MONITOR_EVAL_EPISODES,
        )


def _copy_aggregates_to_top_level() -> None:
    mapping = {
        "aggregated_main_results.csv": "second_pass_aggregated_main_results.csv",
        "aggregated_monitor_results.csv": "second_pass_aggregated_monitor_results.csv",
        "aggregated_tampering_results.csv": "second_pass_aggregated_tampering_results.csv",
        "aggregated_resource_results.csv": "second_pass_aggregated_resource_results.csv",
    }
    for source_name, target_name in mapping.items():
        source = MAIN_RESULTS_ROOT / source_name
        if source.exists():
            shutil.copy2(source, TOP_LEVEL_RESULTS / target_name)


def main() -> None:
    profile_name = _profile_name()
    started = time.time()
    summary = {
        "stage": "second_pass",
        "status": "running",
        "profile_name": profile_name,
        "main_results_root": str(MAIN_RESULTS_ROOT),
        "monitor_results_root": str(MONITOR_RESULTS_ROOT),
        "main_eval_episodes": MAIN_EVAL_EPISODES,
        "monitor_eval_episodes": MONITOR_EVAL_EPISODES,
        "gem_timeout_steps": GEM_TIMEOUT_STEPS,
        "reused_checkpoint_envs": ["ProxyFarming-v0", "SensorTamper-v0"],
        "retrained_envs": ["GemTimeout-v0"],
        "monitor_methods": ["PROXY_PPO"],
        "started_at": time.time(),
    }
    _write_summary(summary)
    try:
        _check_guard(profile_name)

        _evaluate_existing_env("ProxyFarming-v0", deterministic=True, results_root=MAIN_RESULTS_ROOT, eval_episodes=MAIN_EVAL_EPISODES, max_saved_per_split=0)
        _evaluate_existing_env("SensorTamper-v0", deterministic=True, results_root=MAIN_RESULTS_ROOT, eval_episodes=MAIN_EVAL_EPISODES, max_saved_per_split=0)
        _train_and_evaluate_gem_timeout(profile_name)

        for env_name in ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"]:
            _check_guard(profile_name)
            _evaluate_proxy_monitor_rollouts(env_name)
            run_monitor_pipeline(
                env_name,
                results_root=MONITOR_RESULTS_ROOT,
                eval_split="id",
                methods=["PROXY_PPO"],
            )

        build_tables(
            results_root=MAIN_RESULTS_ROOT,
            latex_root=ROOT / "latex",
            monitor_results_root=MONITOR_RESULTS_ROOT,
        )
        build_plots(
            results_root=MAIN_RESULTS_ROOT,
            latex_root=ROOT / "latex",
            monitor_results_root=MONITOR_RESULTS_ROOT,
        )
        build_paper_assets(results_root=MAIN_RESULTS_ROOT, latex_root=ROOT / "latex")
        _copy_aggregates_to_top_level()

        summary.update(
            {
                "status": "completed",
                "wall_clock_sec": time.time() - started,
            }
        )
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "stopped",
                "reason": str(exc),
                "wall_clock_sec": time.time() - started,
            }
        )
        _write_summary(summary)
        raise
    _write_summary(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
