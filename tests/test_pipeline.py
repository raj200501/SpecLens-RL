"""End-to-end: the smoke config runs every stage and the report builds from its CSVs."""

from pathlib import Path

import pandas as pd

from speclens.experiment import collect, execute_run, fit_hack_monitors, load_config, plan_penalty_runs, plan_runs
from speclens.report import build

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_pipeline(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["envs"] = ["ProxyFarming"]
    out = tmp_path / "results"
    runs = plan_runs(cfg)
    assert len({r.run_id for r in runs}) == len(runs)
    for run in runs:
        execute_run(run, cfg, out)
    assert execute_run(runs[0], cfg, out).startswith("cached")  # resumable
    collect(cfg, out)

    episodes = pd.read_csv(out / "episodes.csv.gz")
    n_eval = cfg["evaluation"]["episodes"]
    assert len(episodes) == len(runs) * 2 * n_eval
    # Every policy is evaluated on the same layouts (paired comparison).
    assert episodes.groupby(["seed", "method", "channel_scale"]).layout_seed.nunique().eq(2 * n_eval).all()

    # Monitor-penalty stage: fit the monitor on the main results, then train against it.
    fit_hack_monitors(cfg, out)
    penalty_runs = plan_penalty_runs(cfg)
    assert {r.seed for r in penalty_runs}.isdisjoint({r.seed for r in runs})
    for run in penalty_runs:
        execute_run(run, cfg, out)
    collect(cfg, out, penalty_runs, prefix="penalty_")
    penalty = pd.read_csv(out / "penalty_episodes.csv.gz")
    assert {"monitor_score", "flagged"} <= set(penalty.columns)

    paper = tmp_path / "paper"
    numbers = build(cfg, out, paper)
    for name in ("tasks", "training_curves", "dose_response", "monitors", "monitor_penalty"):
        assert (paper / "figures" / f"{name}.pdf").exists()
    for name in ("table_tasks", "table_main", "table_monitors", "table_penalty", "numbers"):
        assert (paper / "generated" / f"{name}.tex").exists()
    assert "ProxyFarming-proxy-id-hack" in numbers
