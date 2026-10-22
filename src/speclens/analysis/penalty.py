"""Analysis of agents trained against a hack monitor."""

from __future__ import annotations

import pandas as pd

from speclens.analysis.stats import summarize
from speclens.monitors.evaluate import build_model
from speclens.monitors.features import channel_features

PENALTY_METRICS = ["hack", "flagged", "undetected_hack", "success", "true_return", "proxy_return"]


def baseline_recall(monitor_data: pd.DataFrame, channel: str, threshold: float = 0.5) -> pd.DataFrame:
    """Out-of-run recall of the monitor on hacking episodes of *unpenalized* final policies.

    For each held-out main run, a monitor fitted on the other runs scores the
    held-out run's final-snapshot ID episodes; recall is the fraction of its
    hacking episodes scored at or above ``threshold``. This is the detection
    rate before any optimization pressure is applied against the monitor.
    """
    rows = []
    names = channel_features(channel)
    for env, data in monitor_data[monitor_data.split == "id"].groupby("env", sort=False):
        final_step = data.checkpoint.max()
        for seed in sorted(data.seed.unique()):
            train = data[data.seed != seed]
            test = data[(data.seed == seed) & (data.checkpoint == final_step) & (data.hack == 1)]
            if test.empty or train.hack.nunique() < 2:
                continue
            clf = build_model("logistic").fit(train[names].to_numpy(), train.hack.astype(int))
            flagged = clf.predict_proba(test[names].to_numpy())[:, 1] >= threshold
            rows.append({"env": env, "seed": seed, "recall": float(flagged.mean())})
    return pd.DataFrame(rows, columns=["env", "seed", "recall"])


def penalized_recall(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per-run recall of the training monitor on hacking episodes of penalized policies."""
    hacks = episodes[(episodes.split == "id") & (episodes.hack == 1)]
    return hacks.groupby(["env", "seed"], as_index=False).flagged.mean().rename(columns={"flagged": "recall"})


def summarize_penalty(episodes: pd.DataFrame, monitor_data: pd.DataFrame, channel: str) -> pd.DataFrame:
    """One row per task: outcomes of penalized agents and monitor recall before/after."""
    id_episodes = episodes[episodes.split == "id"].assign(undetected_hack=lambda d: d.hack * (1 - d.flagged))
    summary = summarize(id_episodes, ["env"], PENALTY_METRICS)
    recalls = {"recall_after": penalized_recall(episodes), "recall_before": baseline_recall(monitor_data, channel)}
    for name, recall in recalls.items():
        stats = summarize(recall, ["env"], ["recall"]).rename(
            columns={"recall": name, "recall_lo": f"{name}_lo", "recall_hi": f"{name}_hi", "n_runs": f"{name}_runs"}
        )
        summary = summary.merge(stats, on="env", how="left")
    return summary
