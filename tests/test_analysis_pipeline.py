import json
from pathlib import Path

import pandas as pd

from src.analysis.aggregate import aggregate_episode_metrics, aggregate_monitor_results, aggregate_run_summaries
from src.analysis.paper_exports import write_experiment_section, write_results_section
from src.analysis.plots import proxy_vs_true_scatter
from src.analysis.tables import dataframe_to_table, environment_summary_table
from src.utils.io_utils import write_csv_gz, write_json


def test_analysis_pipeline_smoke(tmp_path: Path):
    results_root = tmp_path / "results"
    run_dir = results_root / "runs" / "ProxyFarming-v0" / "PROXY_PPO" / "seed_0"
    run_dir.mkdir(parents=True)
    write_csv_gz(
        run_dir / "episodes.csv.gz",
        [
            {
                "env_name": "ProxyFarming-v0",
                "method": "PROXY_PPO",
                "seed": 0,
                "split": "id",
                "proxy_return": 1.0,
                "true_score": 0.0,
                "success": 0,
                "reward_hack": 1,
                "hack_score": 3.0,
            }
        ],
    )
    write_json(run_dir / "summary.json", {"env_name": "ProxyFarming-v0", "method": "PROXY_PPO", "seed": 0, "run_dir": str(run_dir), "total_steps": 1000, "wall_clock_sec": 1.0})
    (results_root / "monitors").mkdir(parents=True)
    (results_root / "monitors" / "ProxyFarming-v0.json").write_text(
        json.dumps([{"env": "ProxyFarming-v0", "channel": "restricted", "model": "logistic_regression", "target": "hack", "ROC_AUC": 0.7, "PR_AUC": 0.6, "leakage_status": "pass"}]),
        encoding="utf-8",
    )
    episodes = aggregate_episode_metrics(results_root)
    runs = aggregate_run_summaries(results_root)
    monitors = aggregate_monitor_results(results_root)
    assert len(episodes) == 1
    assert len(runs) == 1
    assert len(monitors) == 1
    latex_root = tmp_path / "latex"
    (latex_root / "sections").mkdir(parents=True)
    (latex_root / "tables").mkdir(parents=True)
    environment_summary_table(["ProxyFarming-v0"], latex_root / "tables" / "environment_summary.tex")
    dataframe_to_table(pd.DataFrame([{"a": 1.0, "b": 2.0}]), latex_root / "tables" / "toy.tex", "Toy", "tab:toy")
    proxy_vs_true_scatter(episodes, latex_root / "figures" / "scatter.pdf")
    pd.DataFrame([{"env": "ProxyFarming-v0", "method": "PROXY_PPO", "true_score_ID": 0.0, "true_score_OOD": 0.0, "hacking_incidence_ID": 1.0, "hacking_incidence_OOD": 1.0}]).to_csv(results_root / "aggregated_main_results.csv", index=False)
    pd.DataFrame([{"env": "ProxyFarming-v0", "method": "PROXY_PPO", "seeds": 1, "step_budget": 1000, "wall_clock": 1.0, "disk_footprint": 10}]).to_csv(results_root / "aggregated_resource_results.csv", index=False)
    write_experiment_section(results_root, latex_root)
    write_results_section(results_root, latex_root)
    assert (latex_root / "sections" / "experiments_placeholders.tex").exists()
    assert (latex_root / "sections" / "results_placeholders.tex").exists()

