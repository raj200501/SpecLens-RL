#!/usr/bin/env python3
"""Generate LaTeX tables from saved artifacts."""

from __future__ import annotations

import pandas as pd

from _bootstrap import ROOT
from src.analysis.aggregate import aggregate_episode_metrics, aggregate_monitor_results, aggregate_run_summaries
from src.analysis.estimators import summarize_main_metrics, tampering_metrics, threshold_robustness
from src.analysis.tables import dataframe_to_table, environment_summary_table
from src.utils.budget import directory_size_bytes
from src.utils.io_utils import write_json


def build_tables() -> None:
    results_root = ROOT / "results"
    latex_tables = ROOT / "latex" / "tables"
    environment_summary_table(["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"], latex_tables / "environment_summary.tex")
    episodes = aggregate_episode_metrics(results_root)
    run_summaries = aggregate_run_summaries(results_root)
    if not run_summaries.empty:
        resource_df = run_summaries.copy()
        resource_df["disk_footprint"] = resource_df["run_dir"].apply(directory_size_bytes)
        resource_df = (
            resource_df.groupby(["env_name", "method"], as_index=False)
            .agg(
                seeds=("seed", "count"),
                step_budget=("total_steps", "first"),
                wall_clock=("wall_clock_sec", "sum"),
                disk_footprint=("disk_footprint", "sum"),
            )
        )
        resource_df.columns = ["env", "method", "seeds", "step_budget", "wall_clock", "disk_footprint"]
        dataframe_to_table(resource_df, latex_tables / "resource_results.tex", "Resource usage by run.", "tab:resource-results")
        resource_df.to_csv(results_root / "aggregated_resource_results.csv", index=False)
    if episodes.empty:
        return
    main_df = summarize_main_metrics(episodes)
    dataframe_to_table(main_df, latex_tables / "main_specification_results.tex", "Main specification-diagnostic results.", "tab:main-results")
    main_df.to_csv(results_root / "aggregated_main_results.csv", index=False)
    sensor = episodes[episodes["env_name"] == "SensorTamper-v0"]
    tamper_rows = []
    for method, group in sensor.groupby("method"):
        row = {"env": "SensorTamper-v0", "method": method}
        row.update(tampering_metrics(group))
        tamper_rows.append(row)
    tamper_df = pd.DataFrame(tamper_rows)
    if not tamper_df.empty:
        dataframe_to_table(tamper_df, latex_tables / "tampering_results.tex", "Tampering metrics for SensorTamper-v0.", "tab:tampering-results")
        tamper_df.to_csv(results_root / "aggregated_tampering_results.csv", index=False)
    robustness_rows = []
    for (env_name, method), group in episodes.groupby(["env_name", "method"]):
        for row in threshold_robustness(group, thresholds=[1.0, 2.0, 3.0]):
            robustness_rows.append({"env": env_name, "method": method, **row})
    robustness_df = pd.DataFrame(robustness_rows)
    if not robustness_df.empty:
        dataframe_to_table(robustness_df, latex_tables / "threshold_robustness.tex", "Threshold sensitivity for hack incidence.", "tab:threshold-robustness")
    monitor_df = aggregate_monitor_results(results_root)
    if not monitor_df.empty:
        hack = monitor_df[monitor_df["target"] == "hack"].copy()
        pivot = hack.pivot_table(index=["env", "model"], columns="channel", values="ROC_AUC").reset_index()
        if {"restricted", "rich"}.issubset(pivot.columns):
            pivot["monitorability_gap"] = pivot["rich"] - pivot["restricted"]
            merged = hack.merge(pivot[["env", "model", "monitorability_gap"]], on=["env", "model"], how="left")
        else:
            merged = hack.copy()
            merged["monitorability_gap"] = None
        merged = merged.rename(columns={"ROC_AUC": "ROC_AUC_hack", "PR_AUC": "PR_AUC_hack"})
        merged = merged[["env", "channel", "model", "ROC_AUC_hack", "PR_AUC_hack", "monitorability_gap", "leakage_status"]]
        dataframe_to_table(merged, latex_tables / "monitor_results.tex", "Monitor performance on hack detection.", "tab:monitor-results")
        merged.to_csv(results_root / "aggregated_monitor_results.csv", index=False)
        write_json(results_root / "monitor_results_summary.json", {"rows": len(merged)})


def main() -> None:
    build_tables()


if __name__ == "__main__":
    main()
