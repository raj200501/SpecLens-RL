"""Run every experiment in the paper from a single config file.

    python -m speclens.experiment --config configs/paper.yaml --workers 4

Each training run is an independent job: train PPO, evaluate the final policy
on fresh ID and OOD layouts, evaluate intermediate snapshots for training
curves and, for proxy runs, roll out snapshots stochastically to build the
monitor dataset. Job outputs land in ``<out>/runs/<run_id>/`` and are skipped
if already present, so an interrupted sweep resumes where it stopped. The
``collect`` step concatenates job outputs into the compact CSVs under
``<out>/`` that every figure, table and number in the paper is computed from.

The ``penalty`` stage depends on the main results: it fits a hack monitor per
task on the main monitor dataset and trains fresh Proxy PPO agents against it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from speclens import __version__
from speclens.envs import make_env
from speclens.evaluation import rollout
from speclens.monitors.features import extract_features
from speclens.monitors.penalty import LinearMonitor, MonitorPenalty
from speclens.rl import ActorCritic, PPOConfig, train_ppo

# Disjoint layout-seed ranges, so no two evaluations share layouts by accident
# and every policy is evaluated on the same layouts (a paired comparison).
EVAL_SEEDS = {"id": 1_000_000, "ood": 2_000_000}
CURVE_SEED = 3_000_000
MONITOR_SEED = 4_000_000

EPISODE_COLUMNS = ["proxy_return", "true_return", "channel_return", "exploit_events", "success", "hack", "length"]


@dataclass(frozen=True)
class Run:
    env: str
    method: str  # "proxy" (trained on R_p) or "oracle" (trained on J_t)
    seed: int
    channel_scale: float = 1.0
    collect_monitor: bool = False
    monitor_beta: float = 0.0  # > 0: trained against a hack monitor with this penalty weight

    @property
    def run_id(self) -> str:
        scale = "" if self.channel_scale == 1.0 else f"-c{self.channel_scale:g}"
        penalty = f"-mon{self.monitor_beta:g}" if self.monitor_beta else ""
        return f"{self.env}/{self.method}{scale}{penalty}-s{self.seed}"

    def meta(self) -> dict:
        meta = {"env": self.env, "method": self.method, "seed": self.seed, "channel_scale": self.channel_scale}
        if self.monitor_beta:
            meta["monitor_beta"] = self.monitor_beta
        return meta


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def plan_runs(cfg: dict) -> list[Run]:
    """All training runs implied by a config, main matrix first."""
    runs = [
        Run(env, method, seed, collect_monitor=(method == "proxy"))
        for env in cfg["envs"]
        for method in cfg["main"]["methods"]
        for seed in range(cfg["main"]["seeds"])
    ]
    for env in cfg["envs"]:
        for scale in cfg["sweep"]["channel_scales"]:
            for seed in range(cfg["sweep"]["seeds"]):
                run = Run(env, "proxy", seed, channel_scale=float(scale))
                if run.run_id not in {r.run_id for r in runs}:
                    runs.append(run)
    return runs


def plan_penalty_runs(cfg: dict) -> list[Run]:
    """Proxy PPO trained against a hack monitor, on seeds disjoint from the main runs."""
    pen = cfg["monitor_penalty"]
    return [
        Run(env, "proxy", pen["seed_offset"] + seed, monitor_beta=float(pen["beta"]))
        for env in cfg["envs"]
        for seed in range(pen["seeds"])
    ]


def monitor_path(out: Path, env: str) -> Path:
    return out / "hack_monitors" / f"{env}.json"


def fit_hack_monitors(cfg: dict, out: Path) -> None:
    """Fit the monitor each penalty run trains against, on all main-run ID episodes."""
    data = pd.read_csv(out / "monitor_dataset.csv.gz")
    for env in cfg["envs"]:
        part = data[(data.env == env) & (data.split == "id")]
        LinearMonitor.fit(part, cfg["monitor_penalty"]["channel"]).save(monitor_path(out, env))


def checkpoint_schedule(total_steps: int, n: int) -> list[int]:
    return [round(total_steps * (k + 1) / n) for k in range(n)]


def execute_run(run: Run, cfg: dict, out: Path) -> str:
    """Train, evaluate and (optionally) collect monitor data for one run."""
    torch.set_num_threads(1)
    run_dir = out / "runs" / run.run_id
    if (run_dir / "done.json").exists():
        return f"cached {run.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo = PPOConfig(**cfg["ppo"])
    schedule = checkpoint_schedule(ppo.total_steps, cfg["main"]["checkpoints"])

    monitor = LinearMonitor.load(monitor_path(out, run.env)) if run.monitor_beta else None

    def env_fn():
        if monitor is None:
            return make_env(run.env, split="id", channel_scale=run.channel_scale)
        env = make_env(run.env, split="id", channel_scale=run.channel_scale, record_trajectory=True)
        return MonitorPenalty(env, monitor, run.monitor_beta)

    reward = "proxy" if run.method == "proxy" else "true"
    result = train_ppo(env_fn, reward, ppo, seed=run.seed, checkpoint_steps=schedule)
    torch.save(
        {"model_config": result.model.config(), "state_dict": result.model.state_dict(), "run": run.meta()},
        run_dir / "final.pt",
    )
    pd.DataFrame(result.log).assign(**run.meta()).to_csv(run_dir / "train_log.csv", index=False)

    # Final policy: sampled actions (the policy as trained) on fresh layouts from both splits.
    n_eval = cfg["evaluation"]["episodes"]
    episodes = []
    for split, base in EVAL_SEEDS.items():
        records = _evaluate(result.model, run, range(base, base + n_eval), split, base, monitor)
        episodes += [{**run.meta(), **r} for r in records]
    pd.DataFrame(episodes).to_csv(run_dir / "episodes.csv", index=False)

    # Training curve: every snapshot on the same fresh ID layouts.
    n_curve = cfg["evaluation"]["curve_episodes"]
    model = ActorCritic(**result.model.config())
    columns = EPISODE_COLUMNS + (["monitor_score", "flagged"] if monitor else [])
    curve = []
    for step, state in result.checkpoints:
        model.load_state_dict(state)
        records = _evaluate(model, run, range(CURVE_SEED, CURVE_SEED + n_curve), "id", CURVE_SEED, monitor)
        curve.append({**run.meta(), "env_steps": step, **{k: float(np.mean([r[k] for r in records])) for k in columns}})
    pd.DataFrame(curve).to_csv(run_dir / "curve.csv", index=False)

    if run.collect_monitor:
        _collect_monitor_rollouts(run, cfg, result.checkpoints, model, run_dir)

    (run_dir / "done.json").write_text(
        json.dumps({**run.meta(), "wall_clock_sec": result.wall_clock_sec, "ppo": ppo.as_dict()}, indent=2),
        encoding="utf-8",
    )
    return f"trained {run.run_id} in {result.wall_clock_sec:.0f}s"


def _evaluate(model, run: Run, layouts, split: str, action_seed: int, monitor: LinearMonitor | None) -> list[dict]:
    """Roll out on fixed layouts; with a monitor, also record its score and verdict."""
    records = rollout(
        model,
        run.env,
        layouts,
        split=split,
        channel_scale=run.channel_scale,
        record_trajectories=monitor is not None,
        action_seed=action_seed,
    )
    if monitor is not None:
        for r in records:
            r["monitor_score"] = monitor.score_trajectory(r.pop("trajectory"))
            r["flagged"] = int(r["monitor_score"] >= monitor.threshold)
    return records


def _collect_monitor_rollouts(run: Run, cfg: dict, checkpoints, model: ActorCritic, run_dir: Path) -> None:
    """Stochastic rollouts of the policy at several points in training.

    Hacking emerges during training, so pooling snapshots gives monitor
    datasets with both classes, drawn from a single policy lineage per run.
    """
    mon = cfg["monitor"]
    n = mon["episodes_per_checkpoint"]
    rows = []
    for k, (step, state) in enumerate(checkpoints):
        if (k + 1) % mon["checkpoint_stride"]:
            continue
        model.load_state_dict(state)
        for s, split in enumerate(("id", "ood")):
            base = MONITOR_SEED + run.seed * 100_000 + k * 1_000 + s * 500
            records = rollout(
                model,
                run.env,
                range(base, base + n),
                split=split,
                record_trajectories=True,
                action_seed=base,
            )
            for r in records:
                traj = r.pop("trajectory")
                rows.append({**run.meta(), "env_steps": step, "checkpoint": k, **r, **extract_features(traj)})
    pd.DataFrame(rows).to_csv(run_dir / "monitor.csv", index=False)


def collect(cfg: dict, out: Path, runs: list[Run] | None = None, prefix: str = "") -> None:
    """Concatenate per-run outputs into the compact result files."""
    runs = plan_runs(cfg) if runs is None else runs
    parts: dict[str, list[pd.DataFrame]] = {"episodes": [], "curves": [], "train_log": [], "monitor_dataset": []}
    meta = []
    for run in runs:
        run_dir = out / "runs" / run.run_id
        parts["episodes"].append(pd.read_csv(run_dir / "episodes.csv"))
        parts["curves"].append(pd.read_csv(run_dir / "curve.csv"))
        parts["train_log"].append(pd.read_csv(run_dir / "train_log.csv"))
        if run.collect_monitor:
            parts["monitor_dataset"].append(pd.read_csv(run_dir / "monitor.csv"))
        meta.append(json.loads((run_dir / "done.json").read_text(encoding="utf-8")))
    for name, frames in parts.items():
        if frames:
            suffix = ".csv" if name == "curves" else ".csv.gz"
            path = out / f"{prefix}{name}{suffix}"
            pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format="%.6g")
    pd.DataFrame(meta).drop(columns="ppo").to_csv(out / f"{prefix}runs.csv", index=False, float_format="%.4g")
    environment = {
        "speclens": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--only", choices=["main", "penalty"], help="run a single stage")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    if args.only != "penalty":
        _run_all(plan_runs(cfg), cfg, out, args.workers)
        collect(cfg, out)
        print(f"collected main results into {out}/", flush=True)
    if args.only != "main" and "monitor_penalty" in cfg:
        fit_hack_monitors(cfg, out)
        runs = plan_penalty_runs(cfg)
        _run_all(runs, cfg, out, args.workers)
        collect(cfg, out, runs, prefix="penalty_")
        print(f"collected monitor-penalty results into {out}/", flush=True)


def _run_all(runs: list[Run], cfg: dict, out: Path, workers: int) -> None:
    print(f"{len(runs)} runs -> {out}/runs (workers={workers})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(execute_run, runs, [cfg] * len(runs), [out] * len(runs)):
            print(message, flush=True)


if __name__ == "__main__":
    main()
