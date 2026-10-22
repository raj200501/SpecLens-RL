"""End-to-end: the smoke config runs every stage and the report builds from its CSVs."""

from pathlib import Path

import pandas as pd

from speclens.experiment import collect, execute_run, load_config, plan_runs
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

    paper = tmp_path / "paper"
    numbers = build(cfg, out, paper)
    for name in ("tasks", "training_curves", "dose_response", "monitors"):
        assert (paper / "figures" / f"{name}.pdf").exists()
    for name in ("table_tasks", "table_main", "table_monitors", "numbers"):
        assert (paper / "generated" / f"{name}.tex").exists()
    assert "ProxyFarming-proxy-id-hack" in numbers
