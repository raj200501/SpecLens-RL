"""Build every figure, table and number in the paper from ``results/``.

    python -m speclens.report --config configs/paper.yaml

Reads only the compact CSVs written by :mod:`speclens.experiment`, so it runs
in under a minute and does not need the trained policies. Every number the
paper text cites is written to ``paper/generated/numbers.tex`` as a ``\\res{key}``
macro (and to ``results/numbers.json``).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from speclens.analysis import figures, tables
from speclens.analysis.monitoring import analyze
from speclens.analysis.optimal import optimal_behavior
from speclens.analysis.penalty import summarize_penalty
from speclens.analysis.stats import summarize
from speclens.experiment import EVAL_SEEDS, load_config
from speclens.monitors.features import CHANNELS, channel_features
from speclens.rl import PPOConfig

MAIN_METRICS = ["success", "hack", "harmful_hack", "true_return", "proxy_return", "exploit_events", "length"]


def dominance(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per env/split: the worst oracle run's true return minus the best proxy run's."""
    runs = episodes.groupby(["env", "split", "method", "seed"], as_index=False)["true_return"].mean()
    rows = []
    for (env, split), part in runs.groupby(["env", "split"]):
        best_proxy = part[part.method == "proxy"].true_return.max()
        worst_oracle = part[part.method == "oracle"].true_return.min()
        rows.append(
            {"env": env, "split": split, "best_proxy": best_proxy, "worst_oracle": worst_oracle,
             "margin": worst_oracle - best_proxy}
        )  # fmt: skip
    return pd.DataFrame(rows)


def emergence(curves: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Per run: when the snapshot hack rate first reaches ``threshold``, and the best success before that."""
    rows = []
    for (env, seed), run in curves.groupby(["env", "seed"]):
        run = run.sort_values("env_steps")
        crossed = run[run.hack >= threshold]
        onset = crossed.env_steps.iloc[0] if len(crossed) else float("nan")
        before = run[run.env_steps < onset] if len(crossed) else run
        rows.append(
            {"env": env, "seed": seed, "onset": onset, "peak_success_before": before.success.max(),
             "final_hack": run.hack.iloc[-1], "final_success": run.success.iloc[-1]}
        )  # fmt: skip
    return pd.DataFrame(rows)


def build(cfg: dict, results: Path, paper: Path) -> dict[str, str]:
    episodes = pd.read_csv(results / "episodes.csv.gz")
    curves = pd.read_csv(results / "curves.csv")
    monitor_data = pd.read_csv(results / "monitor_dataset.csv.gz")
    runs = pd.read_csv(results / "runs.csv")
    fig_dir, gen_dir = paper / "figures", paper / "generated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    main = episodes[(episodes.channel_scale == 1.0) & (episodes.seed < cfg["main"]["seeds"])]
    main_summary = summarize(main, ["env", "method", "split"], MAIN_METRICS)
    main_summary.to_csv(results / "summary_main.csv", index=False, float_format="%.4f")
    dom = dominance(main)
    dom.to_csv(results / "summary_dominance.csv", index=False, float_format="%.4f")
    main_curves = curves[(curves.method == "proxy") & (curves.channel_scale == 1.0)]
    onsets = emergence(main_curves[main_curves.seed < cfg["main"]["seeds"]])
    onsets.to_csv(results / "summary_emergence.csv", index=False, float_format="%.4f")

    sweep = episodes[(episodes.method == "proxy") & (episodes.seed < cfg["sweep"]["seeds"])]
    sweep_summary = summarize(sweep, ["env", "channel_scale", "split"], MAIN_METRICS)
    sweep_summary.to_csv(results / "summary_sweep.csv", index=False, float_format="%.4f")
    seeds = range(EVAL_SEEDS["id"], EVAL_SEEDS["id"] + cfg["sweep"]["optimal_layouts"])
    optimal = optimal_behavior(cfg["envs"], cfg["sweep"]["channel_scales"], PPOConfig(**cfg["ppo"]).gamma, seeds)
    optimal_summary = optimal.groupby(["env", "channel_scale"], as_index=False)[MAIN_METRICS].mean()
    optimal_summary.to_csv(results / "summary_optimal.csv", index=False, float_format="%.4f")

    monitor_results, monitor_features = analyze(monitor_data, cfg["monitor"]["models"])
    monitor_results.to_csv(results / "monitor_results.csv", index=False, float_format="%.4f")
    monitor_features.to_csv(results / "monitor_single_features.csv", index=False, float_format="%.4f")

    figures.tasks_figure(fig_dir / "tasks")
    figures.curves_figure(curves, fig_dir / "training_curves")
    figures.dose_response_figure(episodes, optimal_summary, cfg["sweep"]["seeds"], fig_dir / "dose_response")
    figures.monitor_figure(monitor_results, fig_dir / "monitors")

    tables.task_table(gen_dir / "table_tasks.tex")
    tables.spec_table(gen_dir / "table_specs.tex")
    tables.hyperparameter_table(cfg, gen_dir / "table_hyperparameters.tex")
    tables.main_results_table(main_summary, gen_dir / "table_main.tex")
    tables.sweep_table(sweep_summary, gen_dir / "table_sweep.tex")
    for model in cfg["monitor"]["models"]:
        for target in ("hack", "harmful_hack", "failure"):
            suffix = "" if (model, target) == ("logistic", "hack") else f"_{target}_{model}"
            tables.monitor_table(monitor_results, gen_dir / f"table_monitors{suffix}.tex", model=model, target=target)

    numbers = dict(_config_numbers(cfg))
    numbers.update(_result_numbers(cfg, main_summary, dom, sweep_summary, monitor_results, runs, monitor_data))
    numbers.update(_emergence_numbers(onsets, main_curves))
    numbers.update(_optimal_numbers(optimal_summary))
    if (results / "penalty_episodes.csv.gz").exists():
        penalty_episodes = pd.read_csv(results / "penalty_episodes.csv.gz")
        penalty = summarize_penalty(penalty_episodes, monitor_data, cfg["monitor_penalty"]["channel"])
        penalty.to_csv(results / "summary_penalty.csv", index=False, float_format="%.4f")
        figures.penalty_figure(pd.read_csv(results / "penalty_curves.csv"), curves, fig_dir / "monitor_penalty")
        tables.penalty_table(penalty, cfg["monitor_penalty"]["channel"], gen_dir / "table_penalty.tex")
        numbers.update(_penalty_numbers(cfg, penalty, pd.read_csv(results / "penalty_runs.csv")))
    tables.write_numbers(numbers.items(), gen_dir / "numbers.tex")
    (results / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    return numbers


def _pct(value: float) -> str:
    return "--" if pd.isna(value) else f"{100 * value:.0f}"


def _num(value: float, digits: int = 2) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def _config_numbers(cfg: dict) -> Iterator[tuple[str, str]]:
    """Settings the paper text mentions, so prose and config cannot drift apart."""
    ppo = PPOConfig(**cfg["ppo"])
    yield "steps", f"{ppo.total_steps / 1e6:g}M"
    yield "n-envs", str(ppo.n_envs)
    yield "rollout-steps", str(ppo.rollout_steps)
    yield "epochs", str(ppo.epochs)
    yield "minibatch", str(ppo.minibatch_size)
    yield "lr", f"{ppo.learning_rate:g}"
    yield "gamma", f"{ppo.gamma:g}"
    yield "gae-lambda", f"{ppo.gae_lambda:g}"
    yield "entropy", f"{ppo.entropy_coef:g}"
    yield "hidden", str(ppo.hidden)
    yield "channels", str(ppo.channels)
    yield "seeds", str(cfg["main"]["seeds"])
    yield "snapshots", str(cfg["main"]["checkpoints"])
    yield "eval-episodes", str(cfg["evaluation"]["episodes"])
    yield "curve-episodes", str(cfg["evaluation"]["curve_episodes"])
    yield "sweep-seeds", str(cfg["sweep"]["seeds"])
    yield "sweep-scales", ", ".join(f"{s:g}" for s in cfg["sweep"]["channel_scales"])
    yield "optimal-layouts", str(cfg["sweep"]["optimal_layouts"])
    yield "monitor-snapshots", str(cfg["main"]["checkpoints"] // cfg["monitor"]["checkpoint_stride"])
    yield "monitor-layouts", str(cfg["monitor"]["episodes_per_checkpoint"])
    for channel in CHANNELS:
        yield f"n-features-{channel}", str(len(channel_features(channel)))


def _result_numbers(cfg, main_summary, dom, sweep_summary, monitor_results, runs, monitor_data):
    yield "n-runs", str(len(runs))
    yield "cpu-hours", f"{runs.total_sec.sum() / 3600:.1f}"
    per_env = monitor_data[monitor_data.split == "id"].groupby("env").size()
    yield "monitor-episodes", f"{int(per_env.iloc[0]):,}" if per_env.nunique() == 1 else "varies"
    for _, r in main_summary.iterrows():
        key = f"{r.env}-{r.method}-{r.split}"
        for m in ("success", "hack", "harmful_hack"):
            yield f"{key}-{m.replace('_', '-')}", _pct(r[m])
            yield f"{key}-{m.replace('_', '-')}-ci", f"[{_pct(r[m + '_lo'])}, {_pct(r[m + '_hi'])}]"
        for m in ("true_return", "proxy_return"):
            yield f"{key}-{m.split('_')[0]}", _num(r[m])
    for _, r in dom.iterrows():
        yield f"{r.env}-{r.split}-dominance-margin", _num(r.margin)
    yield "dominance-all", "yes" if (dom.margin > 0).all() else "no"
    for _, r in sweep_summary[sweep_summary.split == "id"].iterrows():
        yield f"sweep-{r.env}-{r.channel_scale:g}-hack", _pct(r.hack)
        yield f"sweep-{r.env}-{r.channel_scale:g}-success", _pct(r.success)
    for _, r in monitor_results.dropna(subset=["model"]).iterrows():
        key = f"mon-{r.env}-{r.target}-{r.channel}-{r.model}"
        yield f"{key}-auc", _num(r.auc)
        yield f"{key}-ci", f"[{_num(r.auc_lo)}, {_num(r.auc_hi)}]"
        yield f"{key}-within", _num(r.within_snapshot_auc)
        yield f"{key}-identity", _num(r.identity_auc)
        yield f"{key}-single", _num(r.selected_feature_auc)
        for gain in ("identity", "feature"):
            yield f"{key}-gain-{gain}", _num(r[f"gain_over_{gain}"])
            yield f"{key}-gain-{gain}-ci", f"[{_num(r[f'gain_over_{gain}_lo'])}, {_num(r[f'gain_over_{gain}_hi'])}]"
        yield f"{key}-ood", _num(r.ood_auc)
        yield f"{key}-late", _num(r.late_auc)
    for _, r in monitor_results.drop_duplicates(["env", "target"]).iterrows():
        yield f"mon-{r.env}-{r.target}-rate", _pct(r.positive_rate)


def _steps(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value / 1e6:.2f}M" if value >= 1e6 else f"{value / 1e3:.0f}k"


def _emergence_numbers(onsets, curves):
    yield "first-snapshot", _steps(curves.env_steps.min())
    for env, part in onsets.groupby("env"):
        switched = part.dropna(subset=["onset"])
        yield f"dyn-{env}-runs", str(len(part))
        yield f"dyn-{env}-switched", str(len(switched))
        yield f"dyn-{env}-onset-median", _steps(switched.onset.median())
        yield f"dyn-{env}-onset-min", _steps(switched.onset.min())
        yield f"dyn-{env}-onset-max", _steps(switched.onset.max())
        yield f"dyn-{env}-peak-success-before", _pct(part.peak_success_before.mean())
        yield f"dyn-{env}-peak-success-before-min", _pct(part.peak_success_before.min())


def _optimal_numbers(summary):
    for _, r in summary.iterrows():
        yield f"opt-{r.env}-{r.channel_scale:g}-hack", _pct(r.hack)
        yield f"opt-{r.env}-{r.channel_scale:g}-success", _pct(r.success)


def _penalty_numbers(cfg, penalty, runs):
    yield "pen-seeds", str(cfg["monitor_penalty"]["seeds"])
    yield "pen-beta", f"{cfg['monitor_penalty']['beta']:g}"
    yield "pen-cpu-hours", f"{runs.total_sec.sum() / 3600:.1f}"
    metrics = ("hack", "harmful_hack", "flagged", "undetected_hack", "undetected_harmful_hack", "success",
               "recall_before", "recall_after")  # fmt: skip
    for _, r in penalty.iterrows():
        for m in metrics:
            yield f"pen-{r.env}-{m.replace('_', '-')}", _pct(r[m])


def missing_macros(paper: Path, numbers: dict[str, str]) -> list[str]:
    """``\\res{key}`` references in the paper sources that the report does not define."""
    cited = set()
    for tex in sorted(paper.rglob("*.tex")):
        if "generated" not in tex.parts:
            cited |= set(re.findall(r"\\res\{([^}]*)\}", tex.read_text(encoding="utf-8")))
    return sorted(cited - set(numbers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--results", default="results")
    parser.add_argument("--paper", default="paper")
    args = parser.parse_args()
    numbers = build(load_config(args.config), Path(args.results), Path(args.paper))
    print(f"wrote figures, tables and {len(numbers)} numbers to {args.paper}/")
    if missing := missing_macros(Path(args.paper), numbers):
        raise SystemExit(f"paper cites undefined numbers: {', '.join(missing)}")


if __name__ == "__main__":
    main()
