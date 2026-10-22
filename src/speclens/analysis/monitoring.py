"""Monitorability analysis over the pooled-checkpoint monitor dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from speclens.monitors.evaluate import (
    TRIVIAL_AUC,
    grouped_cv,
    grouped_transfer,
    label_status,
    pr_auc,
    run_bootstrap_ci,
    single_feature_cv,
)
from speclens.monitors.features import CHANNELS, channel_features, uses_label_sources

TARGETS = {"hack": lambda d: d["hack"].astype(int), "failure": lambda d: 1 - d["success"].astype(int)}


def analyze(dataset: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every (env, target, channel, model) monitor.

    Returns ``(results, features)``: one row per monitor with leave-one-run-out
    ROC-AUC (mean over held-out runs, bootstrap CI over runs), ID->OOD and
    early->late transfer AUCs, the best single-feature baseline and the
    admissibility flags; and one row per (env, target, channel, feature) with
    its held-out single-feature AUC.
    """
    rows, feature_rows = [], []
    sort_keys = ["seed", "checkpoint", "split", "layout_seed"]
    for env, data in dataset.groupby("env", sort=False):
        data = data.sort_values(sort_keys).reset_index(drop=True)
        id_rows, ood_rows = data[data.split == "id"], data[data.split == "ood"]
        late_cut = data["checkpoint"].median()
        early, late = id_rows[id_rows.checkpoint <= late_cut], id_rows[id_rows.checkpoint > late_cut]
        for target, label in TARGETS.items():
            y = label(id_rows).to_numpy()
            groups = id_rows["seed"].to_numpy()
            status = label_status(y, groups)
            for channel in CHANNELS:
                cols = channel_features(channel)
                base = {
                    "env": env,
                    "target": target,
                    "channel": channel,
                    "n_episodes": len(y),
                    "n_runs": len(np.unique(groups)),
                    "positive_rate": float(y.mean()),
                    "label_status": status,
                    "leaky_by_construction": uses_label_sources(channel),
                }
                if status != "ok":
                    rows.append({**base, "admissible": False})
                    continue
                X = id_rows[cols].to_numpy()
                single = single_feature_cv(X, y, groups)
                for name, auc in zip(cols, single):
                    feature_rows.append({"env": env, "target": target, "channel": channel, "feature": name, "auc": auc})
                best = int(np.argmax(single))
                for model in models:
                    cv = grouped_cv(X, y, groups, model)
                    lo, hi = run_bootstrap_ci(cv.fold_auc)
                    to_ood = grouped_transfer(
                        X,
                        y,
                        groups,
                        ood_rows[cols].to_numpy(),
                        label(ood_rows).to_numpy(),
                        ood_rows.seed.to_numpy(),
                        model,
                    )
                    to_late = grouped_transfer(
                        early[cols].to_numpy(),
                        label(early).to_numpy(),
                        early.seed.to_numpy(),
                        late[cols].to_numpy(),
                        label(late).to_numpy(),
                        late.seed.to_numpy(),
                        model,
                    )
                    rows.append(
                        {
                            **base,
                            "model": model,
                            "auc": cv.mean_auc,
                            "auc_lo": lo,
                            "auc_hi": hi,
                            "auc_min_fold": float(np.min(cv.fold_auc)),
                            "pr_auc": pr_auc(y, cv.oof_scores),
                            "ood_auc": float(np.mean(to_ood)) if to_ood else np.nan,
                            "late_auc": float(np.mean(to_late)) if to_late else np.nan,
                            "best_feature": cols[best],
                            "best_feature_auc": float(single[best]),
                            "trivial": bool(single[best] >= TRIVIAL_AUC),
                            "admissible": not uses_label_sources(channel),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)
