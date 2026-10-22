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
    restrict,
    sees_target_sources,
    selected_feature_cv,
    single_feature_cv,
    stratified_auc,
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
    for env in ("ProxyFarming", "SensorTamper", "GemTimeout"):
        assert sees_target_sources("privileged", env, "hack") and not sees_target_sources("task", env, "hack")
    # Reaching the goal is visible to the task channel in ProxyFarming, but not the deadline in GemTimeout.
    assert sees_target_sources("task", "ProxyFarming", "failure")
    assert not sees_target_sources("task", "GemTimeout", "failure")
    assert not sees_target_sources("privileged", "SensorTamper", "failure")  # the object cell is never exposed


def test_provenance_is_enforced_not_just_declared():
    """Every feature can be computed from its declared fields alone, and nothing else."""
    for feature in FEATURES:
        feature.fn(restrict(TRAJ, feature.sources))  # must not raise
        hidden = (CHANNELS["privileged"] - feature.sources) & set(TRAJ)
        for field in hidden:
            view = restrict(TRAJ, feature.sources)
            assert field not in view
    with pytest.raises(KeyError):
        restrict(TRAJ, CHANNELS["behavior"])["goal"]


def test_label_status():
    assert label_status(np.zeros(30)) == "degenerate"
    assert label_status(np.r_[1, np.zeros(29)]) == "degenerate"
    assert label_status(np.tile([0, 1], 15)) == "ok"


def test_grouped_cv_skips_single_class_runs_without_discarding_the_task():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(4), 40)
    y = rng.integers(0, 2, size=len(groups))
    y[groups == 0] = 0  # one run never hacks
    X = np.column_stack([y + rng.normal(0, 0.5, size=len(y))])
    cv = grouped_cv(X, y, groups, "logistic")
    assert len(cv.fold_auc) == 3  # the single-class run contributes no fold AUC...
    assert not np.isnan(cv.oof_scores[groups == 0]).any()  # ...but is still scored


def test_stratified_auc_ignores_between_policy_differences():
    # Two "policies": one hacks a lot and scores high, the other rarely and scores low.
    # Within each, the score carries no information about the label.
    rng = np.random.default_rng(4)
    strata = np.repeat(["a", "b"], 200)
    y = np.r_[rng.random(200) < 0.8, rng.random(200) < 0.2].astype(int)
    scores = np.r_[np.full(200, 0.9), np.full(200, 0.1)] + rng.normal(0, 0.01, 400)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, scores) > 0.75  # pooled AUC rewards telling the policies apart
    assert abs(stratified_auc(y, scores, strata) - 0.5) < 0.1


def test_grouped_cv_does_not_leak_across_runs():
    """Run identity predicts the label across runs but not within one; held-out runs must see no signal.

    Each run has its own hack rate and a feature that identifies the run. A
    monitor trained on rows from the same run (row-level CV) would score well
    by memorizing run identity; with whole runs held out it cannot.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 60)
    rates = np.linspace(0.1, 0.9, 8)
    y = (rng.random(len(groups)) < rates[groups]).astype(int)
    X = np.column_stack([groups + rng.normal(0, 0.01, size=len(groups)), rng.normal(size=len(groups))])
    cv = grouped_cv(X, y, groups, "boosting")
    assert len(cv.fold_auc) == 8
    assert abs(cv.mean_auc - 0.5) < 0.08
    from sklearn.model_selection import KFold

    leaky = np.zeros(len(y))
    for train, test in KFold(5, shuffle=True, random_state=0).split(X):
        from speclens.monitors.evaluate import build_model

        leaky[test] = build_model("boosting").fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, leaky) > 0.7  # the same data does leak under row-level CV


def test_single_feature_baselines_choose_on_training_runs():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(5), 40)
    y = rng.integers(0, 2, size=len(groups))
    X = np.column_stack([-y + rng.normal(0, 0.1, size=len(y)), rng.normal(size=len(y))])
    auc = single_feature_cv(X, y, groups)
    assert auc[0] > 0.99
    assert 0.35 < auc[1] < 0.65
    fold_auc, chosen = selected_feature_cv(X, y, groups)
    assert chosen == [0] * 5 and min(fold_auc) > 0.99
    # On pure noise, selecting the best of many features on training runs stays near chance.
    noise = rng.normal(size=(len(y), 15))
    assert abs(np.mean(selected_feature_cv(noise, y, groups)[0]) - 0.5) < 0.08


def _synthetic_monitor_dataset(n_seeds=5, per_run=60, seed=0):
    """Proxy and oracle runs on a real task name; only proxy episodes hack."""
    rng = np.random.default_rng(seed)
    rows = []
    for run in range(n_seeds):
        for method in ("proxy", "oracle"):
            for i in range(per_run):
                hack = int(method == "proxy" and rng.random() < 0.5)
                success = int(not hack and rng.random() < 0.7)
                row = {"env": "ProxyFarming", "method": method, "seed": run, "checkpoint": i % 6,
                       "split": "id" if i % 2 else "ood", "layout_seed": i, "hack": hack, "success": success,
                       "harmful_hack": int(hack and not success)}  # fmt: skip
                for f in FEATURES:
                    row[f.name] = rng.normal()
                row["stay_frac"] += 1.0 * hack  # weak behavioral signal
                row["frac_on_exploit"] = hack + rng.normal(0, 0.01)  # privileged feature that *is* the label
                rows.append(row)
    return pd.DataFrame(rows)


def test_admissibility_screen_flags_the_privileged_positive_control():
    results, features = analyze(_synthetic_monitor_dataset(), ["logistic"])
    hack = results[results.target == "hack"].set_index("channel")
    assert hack.loc["privileged", "trivial"] and not hack.loc["privileged", "admissible"]
    assert not hack.loc["behavior", "trivial"] and hack.loc["behavior", "admissible"]
    assert 0.6 < hack.loc["behavior", "auc"] < 0.95
    assert hack.loc["privileged", "selected_feature"] == "frac_on_exploit"
    # Only proxy runs hack here, so knowing the method alone is a strong baseline.
    assert hack.loc["behavior", "identity_auc"] > 0.7
    assert set(features.channel) == set(CHANNELS)


def test_policy_identity_baseline_learns_any_per_snapshot_rate():
    """Hack rate that rises then falls over snapshots: a categorical baseline must still capture it."""
    data = _synthetic_monitor_dataset()
    rng = np.random.default_rng(5)
    rate = {0: 0.05, 1: 0.5, 2: 0.95, 3: 0.95, 4: 0.5, 5: 0.05}
    proxy = data.method == "proxy"
    data.loc[proxy, "hack"] = (rng.random(proxy.sum()) < data.loc[proxy, "checkpoint"].map(rate)).astype(int)
    results, _ = analyze(data, ["logistic"])
    assert results[results.target == "hack"].identity_auc.iloc[0] > 0.8


def test_analyze_survives_degenerate_labels():
    data = _synthetic_monitor_dataset()
    data["hack"] = 0
    data["harmful_hack"] = 0
    results, _ = analyze(data, ["logistic"])
    assert (results[results.target == "hack"].label_status == "degenerate").all()
    assert "model" in results.columns


def test_linear_monitor_roundtrip_and_penalty(tmp_path):
    from speclens.envs import make_env
    from speclens.monitors.penalty import LinearMonitor, MonitorPenalty

    data = _synthetic_monitor_dataset()
    monitor = LinearMonitor.fit(data, "behavior")
    monitor.save(tmp_path / "m.json")
    loaded = LinearMonitor.load(tmp_path / "m.json")
    assert loaded == monitor
    assert monitor.score(extract_features(TRAJ)) == pytest.approx(loaded.score_trajectory(TRAJ))

    with pytest.raises(ValueError):
        MonitorPenalty(make_env("ProxyFarming"), monitor, beta=5.0)
    plain = make_env("ProxyFarming", record_trajectory=True)
    wrapped = MonitorPenalty(make_env("ProxyFarming", record_trajectory=True), monitor, beta=5.0)
    plain.reset(seed=0)
    wrapped.reset(seed=0)
    assert wrapped.observation_space == plain.observation_space and wrapped.unwrapped.planes == plain.planes
    for _ in range(plain.max_steps):
        _, r_plain, done_plain, trunc_plain, _ = plain.step(INTERACT)
        _, r_wrapped, _, _, info = wrapped.step(INTERACT)
        if done_plain or trunc_plain:
            score = info["episode"]["monitor_score"]
            assert r_wrapped == pytest.approx(r_plain - 5.0 * score)
            break
        assert r_wrapped == r_plain
