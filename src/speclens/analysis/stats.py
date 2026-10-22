"""Aggregation with run-level uncertainty.

Training runs are the independent units: episode outcomes are first averaged
within each run, and uncertainty is a 95% Student-t interval for the mean over
runs. With 5-10 runs, percentile-bootstrap intervals are noticeably
anti-conservative (they cover the mean in roughly 83-90% of repetitions rather
than 95%), so we use the t interval, clipped to the metric's range where it has
one (rates, AUCs).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

UNIT_INTERVAL_PREFIXES = ("success", "hack", "harmful_hack", "flagged", "undetected", "recall", "auc")


def mean_ci(
    values: Sequence[float], alpha: float = 0.05, bounds: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Student-t confidence interval for the mean of independent per-run values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(1 - alpha / 2, df=len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    lo, hi = values.mean() - half, values.mean() + half
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return float(lo), float(hi)


def metric_bounds(metric: str) -> tuple[float, float] | None:
    return (0.0, 1.0) if metric.startswith(UNIT_INTERVAL_PREFIXES) else None


def per_run_means(df: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Average episode metrics within each run (keys + seed)."""
    return df.groupby([*keys, "seed"], as_index=False)[list(metrics)].mean()


def summarize(df: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and 95% CI over runs for each group of ``keys``.

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
            row[f"{m}_lo"], row[f"{m}_hi"] = mean_ci(part[m].to_numpy(), bounds=metric_bounds(m))
        rows.append(row)
    return pd.DataFrame(rows)
