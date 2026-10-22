import numpy as np
import pandas as pd
import pytest

from speclens.analysis.monitoring import analyze
from speclens.envs.base import DOWN, INTERACT, RIGHT
from speclens.monitors import (
    CHANNELS,
    FEATURES,
    channel_features,
    extract_features,
    grouped_cv,
    label_status,
    single_feature_cv,
    uses_label_sources,
)

TRAJ = {
    "positions": [[0, 0], [0, 1], [0, 1], [0, 1], [1, 1]],
    "actions": [RIGHT, INTERACT, INTERACT, DOWN],
    "goal": [3, 3],
    "start": [0, 0],
    "walls": [],
    "exploit_cells": [[0, 1]],
    "size": 7,
    "max_steps": 40,
}


def test_feature_values_on_a_handmade_trajectory():
    f = extract_features(TRAJ)
    assert f["length_ratio"] == pytest.approx(4 / 40)
    assert f["frac_interact"] == pytest.approx(0.5)
    assert f["stay_frac"] == pytest.approx(0.5)
    assert f["longest_stay_ratio"] == pytest.approx(0.5)
    assert f["unique_cell_ratio"] == pytest.approx(3 / 5)
    assert f["frac_on_exploit"] == pytest.approx(3 / 4)
    assert f["interact_on_exploit"] == pytest.approx(2 / 4)
    assert f["exploit_visits"] == pytest.approx(1 / 4)
    assert f["goal_progress"] == pytest.approx((6 - 4) / 6)


def test_channels_only_get_features_they_are_allowed_to_see():
    for channel, allowed in CHANNELS.items():
        names = set(channel_features(channel))
        assert names == {f.name for f in FEATURES if f.sources <= allowed}
    assert set(channel_features("behavior")) < set(channel_features("task")) < set(channel_features("privileged"))
    # A behavior monitor must not depend on where the goal or the exploit is.
    moved = {**TRAJ, "goal": [6, 6], "exploit_cells": [[5, 5]]}
    a, b = extract_features(TRAJ), extract_features(moved)
    assert all(a[n] == b[n] for n in channel_features("behavior"))
    assert uses_label_sources("privileged") and not uses_label_sources("task")


def test_label_status():
    groups = np.repeat([0, 1, 2], 10)
    assert label_status(np.zeros(30), groups) == "degenerate"
    assert label_status(np.r_[1, np.zeros(29)], groups) == "degenerate"
    y = np.tile([0, 1], 15)
    assert label_status(y, groups) == "ok"
    y_one_class_run = y.copy()
    y_one_class_run[:10] = 1
    assert label_status(y_one_class_run, groups) == "single-class-fold"


def test_grouped_cv_does_not_leak_across_runs():
    """A feature that only identifies the run must not look predictive when runs are held out.

    Each run has a constant label rate but its own feature offset; a monitor that
    memorized runs would score highly under row-level CV but not here.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 50)
    y = rng.integers(0, 2, size=len(groups))
    X = (groups[:, None] + rng.normal(0, 0.01, size=(len(groups), 3))).astype(float)
    cv = grouped_cv(X, y, groups, "logistic")
    assert len(cv.fold_auc) == 6
    assert abs(cv.mean_auc - 0.5) < 0.1
    assert not np.isnan(cv.oof_scores).any()


def test_single_feature_baseline_picks_orientation_on_training_runs():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(5), 40)
    y = rng.integers(0, 2, size=len(groups))
    X = np.column_stack([-y + rng.normal(0, 0.1, size=len(y)), rng.normal(size=len(y))])
    auc = single_feature_cv(X, y, groups)
    assert auc[0] > 0.99
    assert 0.35 < auc[1] < 0.65


def _synthetic_monitor_dataset(n_runs=5, per_run=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for run in range(n_runs):
        for i in range(per_run):
            hack = int(rng.random() < 0.4)
            row = {
                "env": "Toy",
                "seed": run,
                "checkpoint": i % 6,
                "split": "id" if i % 2 else "ood",
                "layout_seed": i,
                "hack": hack,
                "success": int(not hack and rng.random() < 0.7),
            }
            for f in FEATURES:
                row[f.name] = rng.normal()
            row["stay_frac"] += 1.0 * hack  # weak behavioral signal
            row["frac_on_exploit"] = hack + rng.normal(0, 0.01)  # privileged feature that *is* the label
            rows.append(row)
    return pd.DataFrame(rows)


def test_leakage_screen_flags_the_privileged_positive_control():
    results, features = analyze(_synthetic_monitor_dataset(), ["logistic"])
    hack = results[results.target == "hack"].set_index("channel")
    assert hack.loc["privileged", "trivial"] and not hack.loc["privileged", "admissible"]
    assert not hack.loc["behavior", "trivial"] and hack.loc["behavior", "admissible"]
    assert 0.6 < hack.loc["behavior", "auc"] < 0.95
    assert hack.loc["privileged", "best_feature"] == "frac_on_exploit"
    assert set(features.channel) == set(CHANNELS)
