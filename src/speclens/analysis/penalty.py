"""Analysis of agents trained against a hack monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from speclens.analysis.stats import summarize
from speclens.monitors.evaluate import build_model
from speclens.monitors.features import channel_features
from speclens.monitors.penalty import LinearMonitor

MIN_RECALL_HACKS = 50  # a run's recall on fewer hacking episodes than this is not reported

PENALTY_METRICS = [
    "hack", "harmful_hack", "flagged", "undetected_hack", "undetected_harmful_hack",
    "success", "true_return", "proxy_return",
]  # fmt: skip


def baseline_recall(monitor_data: pd.DataFrame, channel: str, threshold: float = 0.5) -> pd.DataFrame:
    """Out-of-run recall of the monitor on hacking episodes of *unpenalized* final policies.

    For each held-out seed, a monitor fitted on the other seeds (proxy and
    oracle runs, all snapshots) scores the held-out Proxy PPO run's
    final-snapshot ID episodes; recall is the fraction of its hacking episodes
    scored at or above ``threshold``. This is the detection rate before any
    optimization pressure is applied against the monitor.
    """
    rows = []
    names = channel_features(channel)
    for env, data in monitor_data[monitor_data.split == "id"].groupby("env", sort=False):
        final_snapshot = data.checkpoint.max()
        for seed in sorted(data.seed.unique()):
            train = data[data.seed != seed]
            held_out = data[(data.seed == seed) & (data.method == "proxy")]
            test = held_out[(held_out.checkpoint == final_snapshot) & (held_out.hack == 1)]
            if test.empty or train.hack.nunique() < 2:
                continue
            clf = build_model("logistic").fit(train[names].to_numpy(), train.hack.astype(int))
            flagged = clf.predict_proba(test[names].to_numpy())[:, 1] >= threshold
            rows.append({"env": env, "seed": seed, "recall": float(flagged.mean())})
    return pd.DataFrame(rows, columns=["env", "seed", "recall"])


def penalized_recall(episodes: pd.DataFrame, min_hacks: int = MIN_RECALL_HACKS) -> pd.DataFrame:
    """Per-run recall of the training monitor on hacking episodes of penalized policies.

    Runs with fewer than ``min_hacks`` hacking ID episodes are left out: a
    penalized run that has (almost) stopped hacking has no recall worth
    reporting, and averaging it in would let a handful of incidental hacks
    stand for a run.
    """
    hacks = episodes[(episodes.split == "id") & (episodes.hack == 1)]
    recall = hacks.groupby(["env", "seed"], as_index=False).agg(recall=("flagged", "mean"), n=("flagged", "size"))
    return recall[recall.n >= min_hacks][["env", "seed", "recall"]]


def summarize_penalty(
    episodes: pd.DataFrame, monitor_data: pd.DataFrame, channel: str, min_hacks: int = MIN_RECALL_HACKS
) -> pd.DataFrame:
    """One row per task: outcomes of penalized agents and monitor recall before/after."""
    id_episodes = episodes[episodes.split == "id"].assign(
        undetected_hack=lambda d: d.hack * (1 - d.flagged),
        undetected_harmful_hack=lambda d: d.harmful_hack * (1 - d.flagged),
    )
    summary = summarize(id_episodes, ["env"], PENALTY_METRICS)
    recalls = {
        "recall_after": penalized_recall(episodes, min_hacks),
        "recall_before": baseline_recall(monitor_data, channel),
    }
    for name, recall in recalls.items():
        stats = summarize(recall, ["env"], ["recall"]).rename(
            columns={"recall": name, "recall_lo": f"{name}_lo", "recall_hi": f"{name}_hi", "n_runs": f"{name}_runs"}
        )
        summary = summary.merge(stats, on="env", how="left")
    return summary


def final_hacks(monitor_data: pd.DataFrame) -> pd.DataFrame:
    """Hacking ID episodes of each run's last monitored snapshot."""
    data = monitor_data[(monitor_data.split == "id") & (monitor_data.method == "proxy")]
    return data[(data.checkpoint == data.checkpoint.max()) & (data.hack == 1)]


def evasion(before: pd.DataFrame, after: pd.DataFrame, monitor: LinearMonitor) -> pd.DataFrame:
    """Decompose the change in a linear monitor's score between two sets of hacking episodes.

    The monitor's logit is linear in standardized features, so the change in
    its mean logit from ``before`` (hacks of unpenalized agents) to ``after``
    (hacks of agents trained against the monitor) is exactly the sum over
    features of ``coef * (change in mean) / scale``, where ``scale`` is the
    feature's standard deviation in the monitor's training data. One row per
    feature, with the raw feature means.
    """
    rows = []
    for name, scale, coef in zip(monitor.features, monitor.scale, monitor.coef):
        b, a = float(before[name].mean()), float(after[name].mean())
        rows.append(
            {
                "feature": name,
                "coef": coef,
                "scale": scale,
                "before": b,
                "after": a,
                "contribution": coef * (a - b) / scale,
            }
        )
    return pd.DataFrame(rows)


def monitor_scores(data: pd.DataFrame, monitor: LinearMonitor) -> np.ndarray:
    """The monitor's hack probability for every row (vectorized ``LinearMonitor.score``)."""
    x = (data[list(monitor.features)].to_numpy(dtype=float) - np.array(monitor.mean)) / np.array(monitor.scale)
    return 1.0 / (1.0 + np.exp(-(x @ np.array(monitor.coef) + monitor.intercept)))


def evasion_summary(
    monitor_data: pd.DataFrame,
    penalty_monitor_data: pd.DataFrame,
    monitors: dict[str, LinearMonitor],
    min_hacks: int = 10,
) -> pd.DataFrame:
    """Per task and monitor feature: how the final hacks of penalized agents differ from unpenalized ones.

    Tasks whose penalized agents produce fewer than ``min_hacks`` hacking
    episodes at their last monitored snapshot are left out: there is no hacking
    left to describe. Recall is the fraction of hacks the training monitor flags.
    """
    parts = []
    for env, monitor in monitors.items():
        before = final_hacks(monitor_data[monitor_data.env == env])
        after = final_hacks(penalty_monitor_data[penalty_monitor_data.env == env])
        if len(after) < min_hacks:
            continue
        shift = evasion(before, after, monitor)
        parts.append(
            shift.assign(
                env=env,
                n_before=len(before),
                n_after=len(after),
                runs_after=after.seed.nunique(),
                recall_before=float((monitor_scores(before, monitor) >= monitor.threshold).mean()),
                recall_after=float((monitor_scores(after, monitor) >= monitor.threshold).mean()),
                share=shift.contribution / shift.contribution.sum(),
            )
        )
    columns = ["env", "feature", "coef", "scale", "before", "after", "contribution", "share", "n_before", "n_after",
               "runs_after", "recall_before", "recall_after"]  # fmt: skip
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)
