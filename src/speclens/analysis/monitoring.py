"""Monitorability analysis over the pooled-snapshot monitor dataset.

The dataset holds stochastic rollouts of ten snapshots from every Proxy PPO and
Oracle PPO run. Folds hold out a seed (its proxy and oracle runs together), so
a monitor is always scored on policy lineages it has never seen.

Two baselines guard against monitors that look good for the wrong reason:

* the *policy-identity* baseline sees only which method and which snapshot
  produced an episode. Hacking becomes more common as proxy training goes on,
  so this baseline alone can score well; a monitor must beat it to show that
  it reads behavior rather than training progress;
* the *selected single feature* is the best individual feature, chosen on the
  training runs of each fold. A monitor must beat it to show that learning a
  combination of features adds anything.

Within-snapshot AUC averages the AUC over (run, snapshot) cells that contain
both classes, i.e. it measures whether a monitor separates hacking from
non-hacking episodes of the *same* policy.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from speclens.analysis.stats import mean_ci
from speclens.monitors.evaluate import (
    TRIVIAL_AUC,
    grouped_cv,
    grouped_transfer,
    label_status,
    pr_auc,
    selected_feature_cv,
    single_feature_cv,
    stratified_auc,
)
from speclens.monitors.features import CHANNELS, channel_features, sees_target_sources

TARGETS = {
    "hack": lambda d: d["hack"].astype(int).to_numpy(),
    "harmful_hack": lambda d: d["harmful_hack"].astype(int).to_numpy(),
    "failure": lambda d: 1 - d["success"].astype(int).to_numpy(),
}
RESULT_COLUMNS = [
    "env", "target", "channel", "model", "n_episodes", "n_runs", "positive_rate", "label_status",
    "admissible", "auc", "auc_lo", "auc_hi", "n_folds", "auc_min_fold", "within_snapshot_auc", "pr_auc",
    "ood_auc", "late_auc", "identity_auc", "identity_auc_lo", "identity_auc_hi", "selected_feature",
    "selected_feature_auc", "selected_feature_auc_lo", "selected_feature_auc_hi", "trivial",
    "gain_over_identity", "gain_over_identity_lo", "gain_over_identity_hi",
    "gain_over_feature", "gain_over_feature_lo", "gain_over_feature_hi",
]  # fmt: skip


def policy_identity(data: pd.DataFrame) -> np.ndarray:
    """One-hot (method, snapshot): which kind of policy produced each episode, and nothing else.

    A model on these columns can learn a separate hack rate for every method and
    training stage, which is everything a monitor could learn from policy
    identity alone (short of telling individual seeds apart).
    """
    cells = data["method"] + "/" + data["checkpoint"].astype(str)
    return pd.get_dummies(cells).to_numpy(dtype=float)


def _paired_gain(name: str, monitor_auc: list[float], baseline_auc: list[float]) -> dict[str, float]:
    """Mean per-fold AUC difference with a t interval; folds are the same held-out runs for both."""
    if len(monitor_auc) != len(baseline_auc) or not monitor_auc:
        return {name: np.nan, f"{name}_lo": np.nan, f"{name}_hi": np.nan}
    diff = np.asarray(monitor_auc) - np.asarray(baseline_auc)
    lo, hi = mean_ci(diff)
    return {name: float(diff.mean()), f"{name}_lo": lo, f"{name}_hi": hi}


def analyze(dataset: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every (env, target, channel, model) monitor.

    Returns ``(results, features)``: one row per monitor (columns
    ``RESULT_COLUMNS``) and one row per (env, target, channel, feature) with
    that feature's held-out AUC when used alone.
    """
    rows, feature_rows = [], []
    for env, data in dataset.groupby("env", sort=False):
        data = data.sort_values(["seed", "method", "checkpoint", "split", "layout_seed"]).reset_index(drop=True)
        id_rows, ood_rows = data[data.split == "id"], data[data.split == "ood"]
        cut = id_rows["checkpoint"].median()
        early, late = id_rows[id_rows.checkpoint <= cut], id_rows[id_rows.checkpoint > cut]
        groups = id_rows["seed"].to_numpy()
        strata = id_rows["method"] + "/" + id_rows["seed"].astype(str) + "/" + id_rows["checkpoint"].astype(str)
        strata = strata.to_numpy()
        for target, label in TARGETS.items():
            y = label(id_rows)
            status = label_status(y)
            identity = grouped_cv(policy_identity(id_rows), y, groups, "logistic") if status == "ok" else None
            for channel in CHANNELS:
                row = {
                    "env": env,
                    "target": target,
                    "channel": channel,
                    "n_episodes": len(y),
                    "n_runs": len(np.unique(groups)),
                    "positive_rate": float(y.mean()),
                    "label_status": status,
                    "admissible": status == "ok" and not sees_target_sources(channel, env, target),
                }
                if status != "ok":
                    rows.append(row)
                    continue
                cols = channel_features(channel)
                X = id_rows[cols].to_numpy()
                for name, auc in zip(cols, single_feature_cv(X, y, groups)):
                    feature_rows.append({"env": env, "target": target, "channel": channel, "feature": name, "auc": auc})
                selected_auc, chosen = selected_feature_cv(X, y, groups)
                for model in models:
                    cv = grouped_cv(X, y, groups, model)
                    scored = ~np.isnan(cv.oof_scores)
                    lo, hi = mean_ci(cv.fold_auc, bounds=(0.0, 1.0))
                    to_ood = grouped_transfer(
                        X, y, groups, ood_rows[cols].to_numpy(), label(ood_rows), ood_rows.seed.to_numpy(), model
                    )
                    to_late = grouped_transfer(
                        early[cols].to_numpy(), label(early), early.seed.to_numpy(),
                        late[cols].to_numpy(), label(late), late.seed.to_numpy(), model,
                    )  # fmt: skip
                    rows.append(
                        {
                            **row,
                            "model": model,
                            "auc": cv.mean_auc,
                            "auc_lo": lo,
                            "auc_hi": hi,
                            "n_folds": len(cv.fold_auc),
                            "auc_min_fold": float(np.min(cv.fold_auc)) if cv.fold_auc else np.nan,
                            "within_snapshot_auc": stratified_auc(y[scored], cv.oof_scores[scored], strata[scored]),
                            "pr_auc": pr_auc(y, cv.oof_scores),
                            "ood_auc": float(np.mean(to_ood)) if to_ood else np.nan,
                            "late_auc": float(np.mean(to_late)) if to_late else np.nan,
                            "identity_auc": identity.mean_auc,
                            "identity_auc_lo": mean_ci(identity.fold_auc, bounds=(0.0, 1.0))[0],
                            "identity_auc_hi": mean_ci(identity.fold_auc, bounds=(0.0, 1.0))[1],
                            "selected_feature": cols[Counter(chosen).most_common(1)[0][0]] if chosen else "",
                            "selected_feature_auc": float(np.mean(selected_auc)) if selected_auc else np.nan,
                            "selected_feature_auc_lo": mean_ci(selected_auc, bounds=(0.0, 1.0))[0],
                            "selected_feature_auc_hi": mean_ci(selected_auc, bounds=(0.0, 1.0))[1],
                            "trivial": bool(selected_auc) and float(np.mean(selected_auc)) >= TRIVIAL_AUC,
                            **_paired_gain("gain_over_identity", cv.fold_auc, identity.fold_auc),
                            **_paired_gain("gain_over_feature", cv.fold_auc, selected_auc),
                        }
                    )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS), pd.DataFrame(feature_rows)
