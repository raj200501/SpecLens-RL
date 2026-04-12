#!/usr/bin/env python3
"""Generate artifact-backed figures."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from src.analysis.aggregate import aggregate_episode_metrics, aggregate_monitor_results
from src.analysis.estimators import seed_variability_summary, summarize_main_metrics
from src.analysis.plots import (
    framework_overview,
    generalization_gap,
    hacking_incidence_bars,
    main_results_panel,
    monitorability_gap,
    proxy_vs_true_scatter,
    sensortamper_monitorability,
    tampering_summary,
)


def build_plots(results_root=None, latex_root=None, monitor_results_root=None) -> None:
    results_root = results_root or (ROOT / "results")
    latex_root = latex_root or (ROOT / "latex")
    monitor_results_root = monitor_results_root or results_root
    framework_overview(latex_root / "figures" / "framework_overview.pdf")
    episodes = aggregate_episode_metrics(results_root)
    if episodes.empty:
        return
    main_df = summarize_main_metrics(episodes)
    variability_df = seed_variability_summary(episodes)
    proxy_vs_true_scatter(episodes, latex_root / "figures" / "proxy_vs_true_scatter.pdf")
    hacking_incidence_bars(main_df, latex_root / "figures" / "hacking_incidence_bars.pdf")
    generalization_gap(main_df, latex_root / "figures" / "generalization_gap.pdf")
    main_results_panel(main_df, latex_root / "figures" / "main_results_panel.pdf", variability_df=variability_df)
    tamper_df = episodes[episodes["env_name"] == "SensorTamper-v0"].groupby("method", as_index=False)["tamper_used"].mean()
    if not tamper_df.empty:
        tamper_df = tamper_df.rename(columns={"tamper_used": "tamper_utilization_rate"})
        tampering_summary(tamper_df, latex_root / "figures" / "tampering_summary.pdf")
    monitor_df = aggregate_monitor_results(monitor_results_root)
    if not monitor_df.empty:
        sensortamper_monitorability(monitor_df, latex_root / "figures" / "sensortamper_monitorability.pdf")
        hack_only = monitor_df[monitor_df["target"] == "hack"].copy()
        pivot = hack_only.pivot_table(index=["env", "model"], columns="channel", values="ROC_AUC").reset_index()
        if {"restricted", "rich"}.issubset(pivot.columns):
            pivot["monitorability_gap"] = pivot["rich"] - pivot["restricted"]
        else:
            pivot["monitorability_gap"] = None
        pivot = pivot.merge(
            hack_only.groupby(["env", "model"], as_index=False)["leakage_status"].first(),
            on=["env", "model"],
            how="left",
        )
        monitorability_gap(pivot, latex_root / "figures" / "monitorability_gap.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root")
    parser.add_argument("--monitor-results-root")
    parser.add_argument("--latex-root")
    args = parser.parse_args()
    build_plots(
        results_root=(ROOT / args.results_root) if args.results_root else None,
        latex_root=(ROOT / args.latex_root) if args.latex_root else None,
        monitor_results_root=(ROOT / args.monitor_results_root) if args.monitor_results_root else None,
    )


if __name__ == "__main__":
    main()
