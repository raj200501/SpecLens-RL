"""Aggregate saved run artifacts into analysis data frames."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List

import pandas as pd


def _read_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def aggregate_episode_metrics(results_root: str | Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in Path(results_root).glob("runs/*/*/seed_*/episodes.csv.gz"):
        df = _read_csv_gz(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_run_summaries(results_root: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(results_root).glob("runs/*/*/seed_*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(payload)
    return pd.DataFrame(rows)


def aggregate_monitor_results(results_root: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(results_root).glob("monitors/*.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            rows.extend(payload)
        else:
            rows.append(payload)
    return pd.DataFrame(rows)


def aggregate_monitor_scope(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    monitor_df = aggregate_monitor_results(results_root)
    rows = []
    for path in sorted(results_root.glob("monitor_datasets/*.csv.gz")):
        env_name = path.name.replace(".csv.gz", "")
        dataset = _read_csv_gz(path)
        hack_total = len(dataset)
        hack_positive = int(dataset["label_hack"].sum()) if "label_hack" in dataset.columns else 0
        failure_total = len(dataset)
        failure_positive = int(dataset["label_failure"].sum()) if "label_failure" in dataset.columns else 0
        env_monitor = monitor_df[monitor_df["env"] == env_name].copy() if not monitor_df.empty else pd.DataFrame()
        hack_monitor = env_monitor[env_monitor["target"] == "hack"].copy() if not env_monitor.empty else pd.DataFrame()
        failure_monitor = env_monitor[env_monitor["target"] == "failure"].copy() if not env_monitor.empty else pd.DataFrame()

        def summarize_target(target_df: pd.DataFrame) -> tuple[str, str]:
            if target_df.empty:
                return "missing", ""
            accepted = target_df[target_df["ROC_AUC"].notna()].copy()
            if not accepted.empty:
                channel_order = ["restricted", "rich"]
                channels = [channel for channel in channel_order if channel in set(accepted["channel"])]
                channel_text = " + ".join(channels)
                return "accepted", channel_text
            statuses = sorted(set(target_df["leakage_status"].dropna()))
            if statuses == ["degenerate"]:
                return "degenerate", ""
            if "fail" in statuses:
                return "rejected", ""
            return ",".join(statuses) if statuses else "missing", ""

        hack_status, hack_channels = summarize_target(hack_monitor)
        failure_status, _ = summarize_target(failure_monitor)
        rows.append(
            {
                "env": env_name,
                "hack_positive": hack_positive,
                "hack_total": hack_total,
                "hack_positive_rate": float(hack_positive / hack_total) if hack_total else float("nan"),
                "failure_positive": failure_positive,
                "failure_total": failure_total,
                "failure_positive_rate": float(failure_positive / failure_total) if failure_total else float("nan"),
                "hack_monitor_status": hack_status,
                "hack_monitor_channels": hack_channels,
                "failure_monitor_status": failure_status,
            }
        )
    return pd.DataFrame(rows)
