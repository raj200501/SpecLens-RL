"""Aggregation with seed-level uncertainty.

Training runs are the independent units: episode outcomes are first averaged
within each run, and uncertainty is reported as a percentile bootstrap over
runs (Agarwal et al., 2021 recommend interval estimates over point estimates
for few-seed RL comparisons).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

N_BOOT = 10_000


def bootstrap_ci(
    values: Sequence[float], n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def per_run_means(df: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Average episode metrics within each run (keys + seed)."""
    return df.groupby([*keys, "seed"], as_index=False)[list(metrics)].mean()


def summarize(df: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and 95% bootstrap CI over runs for each group of ``keys``.

    Returns one row per group with columns ``<metric>``, ``<metric>_lo``,
    ``<metric>_hi`` and ``n_runs``.
    """
    runs = per_run_means(df, keys, metrics)
    rows = []
    for group, part in runs.groupby(list(keys), sort=True):
        row = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        row["n_runs"] = len(part)
        for m in metrics:
            row[m] = float(part[m].mean())
            row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_ci(part[m].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)
