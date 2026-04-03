#!/usr/bin/env python3
"""Generate artifact-backed figures."""

from __future__ import annotations

from _bootstrap import ROOT
from src.analysis.aggregate import aggregate_episode_metrics, aggregate_monitor_results
from src.analysis.estimators import summarize_main_metrics
from src.analysis.plots import generalization_gap, hacking_incidence_bars, monitorability_gap, proxy_vs_true_scatter, tampering_summary


def build_plots() -> None:
    episodes = aggregate_episode_metrics(ROOT / "results")
    if episodes.empty:
        return
    main_df = summarize_main_metrics(episodes)
    proxy_vs_true_scatter(episodes, ROOT / "latex" / "figures" / "proxy_vs_true_scatter.pdf")
    hacking_incidence_bars(main_df, ROOT / "latex" / "figures" / "hacking_incidence_bars.pdf")
    generalization_gap(main_df, ROOT / "latex" / "figures" / "generalization_gap.pdf")
    tamper_df = episodes[episodes["env_name"] == "SensorTamper-v0"].groupby("method", as_index=False)["tamper_used"].mean()
    if not tamper_df.empty:
        tamper_df = tamper_df.rename(columns={"tamper_used": "tamper_utilization_rate"})
        tampering_summary(tamper_df, ROOT / "latex" / "figures" / "tampering_summary.pdf")
    monitor_df = aggregate_monitor_results(ROOT / "results")
    if not monitor_df.empty:
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
        monitorability_gap(pivot, ROOT / "latex" / "figures" / "monitorability_gap.pdf")


def main() -> None:
    build_plots()


if __name__ == "__main__":
    main()
