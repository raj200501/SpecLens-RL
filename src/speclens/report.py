"""Build every figure, table and number in the paper from ``results/``.

    python -m speclens.report --config configs/paper.yaml

Reads only the compact CSVs written by :mod:`speclens.experiment`, so it runs
in seconds and does not need the trained policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from speclens.analysis import figures, tables
from speclens.analysis.monitoring import analyze
from speclens.analysis.penalty import summarize_penalty
from speclens.analysis.stats import summarize
from speclens.experiment import load_config

MAIN_METRICS = ["success", "hack", "true_return", "proxy_return", "exploit_events", "length"]


def dominance(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per env/split: does every oracle run beat every proxy run on true return?"""
    runs = episodes.groupby(["env", "split", "method", "seed"], as_index=False)["true_return"].mean()
    rows = []
    for (env, split), part in runs.groupby(["env", "split"]):
        best_proxy = part[part.method == "proxy"].true_return.max()
        worst_oracle = part[part.method == "oracle"].true_return.min()
        rows.append(
            {
                "env": env,
                "split": split,
                "best_proxy": best_proxy,
                "worst_oracle": worst_oracle,
                "margin": worst_oracle - best_proxy,
            }
        )
    return pd.DataFrame(rows)


def build(cfg: dict, results: Path, paper: Path) -> dict:
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

    sweep = episodes[(episodes.method == "proxy") & (episodes.seed < cfg["sweep"]["seeds"])]
    sweep_summary = summarize(sweep, ["env", "channel_scale", "split"], MAIN_METRICS)
    sweep_summary.to_csv(results / "summary_sweep.csv", index=False, float_format="%.4f")

    monitor_results, monitor_features = analyze(monitor_data, cfg["monitor"]["models"])
    monitor_results.to_csv(results / "monitor_results.csv", index=False, float_format="%.4f")
    monitor_features.to_csv(results / "monitor_single_features.csv", index=False, float_format="%.4f")

    figures.tasks_figure(fig_dir / "tasks")
    figures.curves_figure(curves, fig_dir / "training_curves")
    figures.dose_response_figure(episodes, cfg["sweep"]["seeds"], fig_dir / "dose_response")
    figures.monitor_figure(monitor_results, fig_dir / "monitors")

    tables.task_table(gen_dir / "table_tasks.tex")
    tables.main_results_table(main_summary, gen_dir / "table_main.tex")
    tables.monitor_table(monitor_results, gen_dir / "table_monitors.tex")
    tables.monitor_table(monitor_results, gen_dir / "table_monitors_boosting.tex", model="boosting")
    tables.spec_table(gen_dir / "table_specs.tex")
    tables.hyperparameter_table(cfg, gen_dir / "table_hyperparameters.tex")
    tables.sweep_table(sweep_summary, gen_dir / "table_sweep.tex")

    numbers = list(_numbers(cfg, main_summary, dom, sweep_summary, monitor_results, runs, monitor_data))
    if (results / "penalty_episodes.csv.gz").exists():
        penalty = summarize_penalty(
            pd.read_csv(results / "penalty_episodes.csv.gz"), monitor_data, cfg["monitor_penalty"]["channel"]
        )
        penalty.to_csv(results / "summary_penalty.csv", index=False, float_format="%.4f")
        figures.penalty_figure(pd.read_csv(results / "penalty_curves.csv"), curves, fig_dir / "monitor_penalty")
        tables.penalty_table(penalty, gen_dir / "table_penalty.tex")
        numbers += list(_penalty_numbers(cfg, penalty, pd.read_csv(results / "penalty_runs.csv")))
    tables.write_numbers(numbers, gen_dir / "numbers.tex")
    (results / "numbers.json").write_text(json.dumps(dict(numbers), indent=2), encoding="utf-8")
    return dict(numbers)


def _numbers(cfg, main_summary, dom, sweep_summary, monitor_results, runs, monitor_data):
    pct = lambda v: f"{100 * v:.0f}"  # noqa: E731
    yield "seeds", str(cfg["main"]["seeds"])
    yield "sweep-seeds", str(cfg["sweep"]["seeds"])
    yield "steps", f"{cfg['ppo']['total_steps'] / 1e6:g}M"
    yield "eval-episodes", str(cfg["evaluation"]["episodes"])
    yield "n-runs", str(len(runs))
    yield "cpu-hours", f"{runs.wall_clock_sec.sum() / 3600:.1f}"
    yield "monitor-episodes", f"{len(monitor_data[monitor_data.split == 'id']) // monitor_data.env.nunique():,}"
    for _, r in main_summary.iterrows():
        key = f"{r.env}-{r.method}-{r.split}"
        for m in ("success", "hack"):
            yield f"{key}-{m}", pct(r[m])
            yield f"{key}-{m}-ci", f"[{pct(r[m + '_lo'])}, {pct(r[m + '_hi'])}]"
        for m in ("true_return", "proxy_return"):
            yield f"{key}-{m.split('_')[0]}", f"{r[m]:.2f}"
    for _, r in dom.iterrows():
        yield f"{r.env}-{r.split}-dominance-margin", f"{r.margin:.2f}"
    yield "dominance-all", "yes" if (dom.margin > 0).all() else "no"
    for _, r in sweep_summary[sweep_summary.split == "id"].iterrows():
        yield f"sweep-{r.env}-{r.channel_scale:g}-hack", pct(r.hack)
        yield f"sweep-{r.env}-{r.channel_scale:g}-success", pct(r.success)
    for _, r in monitor_results.iterrows():
        if pd.isna(r.get("model")):
            continue
        key = f"mon-{r.env}-{r.target}-{r.channel}-{r.model}"
        yield f"{key}-auc", f"{r.auc:.2f}"
        yield f"{key}-ci", f"[{r.auc_lo:.2f}, {r.auc_hi:.2f}]"
        yield f"{key}-ood", f"{r.ood_auc:.2f}"
        yield f"{key}-late", f"{r.late_auc:.2f}"
        yield f"{key}-single", f"{r.best_feature_auc:.2f}"
    for _, r in monitor_results.drop_duplicates(["env", "target"]).iterrows():
        yield f"mon-{r.env}-{r.target}-rate", pct(r.positive_rate)


def _penalty_numbers(cfg, penalty, runs):
    pct = lambda v: "--" if pd.isna(v) else f"{100 * v:.0f}"  # noqa: E731
    yield "pen-seeds", str(cfg["monitor_penalty"]["seeds"])
    yield "pen-beta", f"{cfg['monitor_penalty']['beta']:g}"
    yield "pen-cpu-hours", f"{runs.wall_clock_sec.sum() / 3600:.1f}"
    for _, r in penalty.iterrows():
        for m in ("hack", "flagged", "undetected_hack", "success", "recall_before", "recall_after"):
            yield f"pen-{r.env}-{m.replace('_', '-')}", pct(r[m])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--results", default="results")
    parser.add_argument("--paper", default="paper")
    args = parser.parse_args()
    numbers = build(load_config(args.config), Path(args.results), Path(args.paper))
    print(f"wrote figures, tables and {len(numbers)} numbers to {args.paper}/")


if __name__ == "__main__":
    main()
