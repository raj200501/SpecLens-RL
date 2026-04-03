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
