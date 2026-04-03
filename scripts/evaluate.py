#!/usr/bin/env python3
"""Evaluate a trained checkpoint on ID and OOD splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from _bootstrap import ROOT
from src.algos import PPOConfig, PPOTrainer
from src.envs import make_env
from src.utils.config import load_config
from src.utils.io_utils import append_inventory, write_csv_gz, write_json
from src.utils.seed import seed_everything
from src.utils.serialization import load_checkpoint


def evaluate_run(
    env_name: str,
    method_name: str,
    seed: int,
    eval_episodes: int,
    results_root: Path | None = None,
    checkpoint_path: Path | None = None,
    deterministic: bool = True,
    max_saved_per_split: int = 5,
    env_kwargs: dict | None = None,
    output_env_name: str | None = None,
) -> Path:
    results_root = Path(results_root) if results_root is not None else (ROOT / "results")
    env_kwargs = env_kwargs or {}
    output_env_name = output_env_name or env_name
    run_dir = results_root / "runs" / output_env_name / method_name / f"seed_{seed}"
    checkpoint = checkpoint_path or (run_dir / "best.ckpt")
    ckpt = load_checkpoint(checkpoint)
    cfg = load_config(
        {
            "PROXY_PPO": "method_proxy_ppo.yaml",
            "ORACLE_TRUE_PPO": "method_oracle_true_ppo.yaml",
            "CONSERVATIVE_PROXY_PPO": "method_conservative_proxy.yaml",
        }[method_name]
    )
    trainer = PPOTrainer(lambda: make_env(env_name, split="id", **env_kwargs), PPOConfig.from_dict(cfg))
    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.model.eval()
    episode_rows = []
    trajectories = []
    start = time.time()
    for split in ["id", "ood"]:
        for episode_idx in range(eval_episodes):
            eval_seed = seed * 1000 + episode_idx + (10000 if split == "ood" else 0)
            seed_everything(eval_seed)
            env = make_env(env_name, split=split, **env_kwargs)
            obs, _ = env.reset(seed=eval_seed)
            done = False
            proxy_return = 0.0
            final_info = {}
            while not done:
                action = trainer.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                proxy_return += float(reward)
                done = terminated or truncated
                final_info = info
            summary = dict(final_info["episode_summary"])
            row = {
                "env_name": output_env_name,
                "base_env_name": env_name,
                "method": method_name,
                "seed": seed,
                "split": split,
                "episode_idx": episode_idx,
                "variant_name": final_info["variant_name"],
                "proxy_return": float(summary.get("proxy_return", proxy_return)),
                "true_score": float(summary.get("true_score", 0.0)),
                "success": int(summary.get("success", False)),
                "reward_hack": int(summary.get("reward_hack", False)),
                "hack_score": float(summary.get("hack_score", 0.0)),
                "episode_length": int(summary.get("episode_length", len(env.trajectory))),
                "tamper_used": int(summary.get("tamper_used", False)),
                "tamper_count": int(summary.get("tamper_count", 0)),
                "beacon_interaction_count": int(summary.get("beacon_interaction_count", 0)),
                "gems_collected": int(summary.get("gems_collected", 0)),
            }
            episode_rows.append(row)
            saved_in_split = sum(1 for record in trajectories if record["split"] == split)
            if max_saved_per_split < 0 or saved_in_split < max_saved_per_split:
                trajectories.append(
                    {
                        "env_name": output_env_name,
                        "base_env_name": env_name,
                        "method": method_name,
                        "seed": seed,
                        "split": split,
                        "deterministic": deterministic,
                        "summary": summary,
                        "context": env.monitor_context(),
                        "trajectory": env.trajectory,
                    }
                )
    write_csv_gz(run_dir / "episodes.csv.gz", episode_rows)
    with (run_dir / "trajectories.json").open("w", encoding="utf-8") as handle:
        json.dump(trajectories, handle)
    eval_summary = {
        "env_name": output_env_name,
        "base_env_name": env_name,
        "method": method_name,
        "seed": seed,
        "eval_episodes_per_split": eval_episodes,
        "episodes_total": len(episode_rows),
        "deterministic": deterministic,
        "saved_trajectories": len(trajectories),
        "wall_clock_sec": time.time() - start,
        "source_checkpoint": str(checkpoint),
        "env_kwargs": env_kwargs,
    }
    write_json(run_dir / "eval_summary.json", eval_summary)
    append_inventory(results_root / "artifact_inventory.csv", run_dir / "episodes.csv.gz", "episode_metrics")
    append_inventory(results_root / "artifact_inventory.csv", run_dir / "trajectories.json", "trajectories")
    append_inventory(results_root / "artifact_inventory.csv", run_dir / "eval_summary.json", "summary")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--results-root")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-saved-per-split", type=int, default=5)
    args = parser.parse_args()
    run_dir = evaluate_run(
        args.env,
        args.method,
        args.seed,
        args.eval_episodes,
        results_root=Path(args.results_root) if args.results_root else None,
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        deterministic=not args.stochastic,
        max_saved_per_split=args.max_saved_per_split,
    )
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
