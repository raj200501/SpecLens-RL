#!/usr/bin/env python3
"""Train a single PPO run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _bootstrap import ROOT
from src.algos import PPOConfig, PPOTrainer
from src.envs import make_env
from src.utils.config import load_config
from src.utils.io_utils import append_inventory, write_csv_gz, write_json
from src.utils.resource_guard import check_resource_guard
from src.utils.seed import seed_everything
from src.utils.serialization import save_checkpoint


def train_run(
    env_name: str,
    method_name: str,
    seed: int,
    total_steps: int,
    profile_name: str,
    results_root: Path | None = None,
    env_kwargs: dict | None = None,
    output_env_name: str | None = None,
) -> Path:
    results_root = Path(results_root) if results_root is not None else (ROOT / "results")
    env_kwargs = env_kwargs or {}
    output_env_name = output_env_name or env_name
    run_dir = results_root / "runs" / output_env_name / method_name / f"seed_{seed}"
    guard = check_resource_guard(ROOT, results_root, profile_name)
    if not guard.allowed:
        raise RuntimeError(guard.reason)
    seed_everything(seed)
    config_file = {
        "PROXY_PPO": "method_proxy_ppo.yaml",
        "ORACLE_TRUE_PPO": "method_oracle_true_ppo.yaml",
        "CONSERVATIVE_PROXY_PPO": "method_conservative_proxy.yaml",
    }[method_name]
    cfg = load_config(config_file)
    trainer = PPOTrainer(lambda: make_env(env_name, split="id", **env_kwargs), PPOConfig.from_dict(cfg))
    start = time.time()
    result = trainer.train(total_steps=total_steps)
    wall_clock = time.time() - start
    checkpoint_payload = {
        "env_name": env_name,
        "method_name": method_name,
        "seed": seed,
        "config": cfg,
        "model_state": trainer.model.state_dict(),
    }
    latest = run_dir / "latest.ckpt"
    best = run_dir / "best.ckpt"
    save_checkpoint(latest, checkpoint_payload)
    save_checkpoint(best, checkpoint_payload)
    episode_rows = []
    for idx, episode in enumerate(result["episode_summaries"]):
        row = {"env_name": env_name, "method": method_name, "seed": seed, "episode_index": idx}
        row.update(episode)
        episode_rows.append(row)
    if episode_rows:
        write_csv_gz(run_dir / "train_metrics.csv.gz", episode_rows)
        append_inventory(results_root / "artifact_inventory.csv", run_dir / "train_metrics.csv.gz", "train_metrics")
    summary = {
        "env_name": env_name,
        "output_env_name": output_env_name,
        "method": method_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "total_steps": total_steps,
        "wall_clock_sec": wall_clock,
        "profile_name": profile_name,
        "episodes_completed": result["episodes"],
        "final_episode": result["final_episode"],
        "env_kwargs": env_kwargs,
    }
    write_json(run_dir / "summary.json", summary)
    append_inventory(results_root / "artifact_inventory.csv", latest, "checkpoint")
    append_inventory(results_root / "artifact_inventory.csv", best, "checkpoint")
    append_inventory(results_root / "artifact_inventory.csv", run_dir / "summary.json", "summary")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--results-root")
    args = parser.parse_args()
    run_dir = train_run(
        args.env,
        args.method,
        args.seed,
        args.steps,
        args.profile,
        results_root=Path(args.results_root) if args.results_root else None,
    )
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
