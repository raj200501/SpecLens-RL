import numpy as np
import pandas as pd
import pytest

from speclens.analysis.penalty import baseline_recall, penalized_recall, summarize_penalty
from speclens.analysis.stats import mean_ci, summarize
from speclens.monitors.features import FEATURES
from speclens.report import dominance


def test_mean_ci_is_a_t_interval_with_valid_coverage():
    values = np.array([0.2, 0.4, 0.6])
    lo, hi = mean_ci(values)
    half = 4.302653 * values.std(ddof=1) / np.sqrt(3)  # t_{0.975, 2}
    assert (lo, hi) == pytest.approx((0.4 - half, 0.4 + half), rel=1e-5)
    assert mean_ci([0.98, 1.0, 1.0], bounds=(0, 1))[1] == 1.0
    assert np.isnan(mean_ci([1.0])[0])
    # Coverage of the true mean in repeated small samples should be close to nominal.
    rng = np.random.default_rng(0)
    covered = [lo <= 0.5 <= hi for lo, hi in (mean_ci(rng.normal(0.5, 0.2, size=5)) for _ in range(2000))]
    assert 0.93 < np.mean(covered) < 0.97


def test_summarize_treats_runs_not_episodes_as_units():
    # Run 0 has 1 episode with value 1; run 1 has 99 episodes with value 0.
    df = pd.DataFrame({"g": ["a"] * 100, "seed": [0] + [1] * 99, "x": [1.0] + [0.0] * 99})
    row = summarize(df, ["g"], ["x"]).iloc[0]
    assert row["x"] == pytest.approx(0.5)  # mean of run means, not of episodes
    assert row["n_runs"] == 2


def test_dominance_margin():
    rows = [
        {"env": "E", "split": "id", "method": m, "seed": s, "true_return": v}
        for m, s, v in [("proxy", 0, 0.1), ("proxy", 1, 0.3), ("oracle", 0, 0.5), ("oracle", 1, 0.9)]
    ]
    margin = dominance(pd.DataFrame(rows)).iloc[0]["margin"]
    assert margin == pytest.approx(0.2)


def test_penalty_recalls():
    episodes = pd.DataFrame(
        {
            "env": "E",
            "seed": [0, 0, 0, 1, 1, 1],
            "split": "id",
            "hack": [1, 1, 0, 1, 0, 0],
            "harmful_hack": [1, 0, 0, 1, 0, 0],
            "flagged": [1, 0, 0, 0, 0, 1],
            "success": [0, 1, 1, 0, 1, 1],
            "true_return": 0.0,
            "proxy_return": 0.0,
        }
    )
    recall = penalized_recall(episodes).set_index("seed").recall
    assert recall.to_dict() == {0: 0.5, 1: 0.0}

    # The unpenalized baseline uses a monitor fitted on other seeds only; a separable
    # feature gives it perfect recall on the held-out final snapshot.
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(3):
        for method in ("proxy", "oracle"):
            for checkpoint in (0, 1):
                for _ in range(20):
                    hack = int(method == "proxy" and rng.random() < 0.5)
                    row = {f.name: rng.normal() for f in FEATURES}
                    row["stay_frac"] = 3.0 * hack
                    rows.append({"env": "E", "seed": seed, "method": method, "checkpoint": checkpoint,
                                 "split": "id", "hack": hack, **row})  # fmt: skip
    before = baseline_recall(pd.DataFrame(rows), "behavior")
    assert len(before) == 3 and (before.recall == 1.0).all()

    # A run-specific signature: each seed's hacks look different, so a monitor that has
    # not seen the seed cannot detect them, while an in-sample monitor would.
    signature_rows = []
    for row in rows:
        row = dict(row)
        row["stay_frac"] = 0.0
        if row["hack"]:
            row[["frac_up", "frac_down", "frac_left"][row["seed"]]] = 5.0
        signature_rows.append(row)
    out_of_run = baseline_recall(pd.DataFrame(signature_rows), "behavior")
    assert (out_of_run.recall < 0.5).all()

    summary = summarize_penalty(episodes, pd.DataFrame(rows), "behavior").iloc[0]
    assert summary.undetected_hack == pytest.approx((1 / 3 + 1 / 3) / 2)
    assert summary.recall_before == 1.0 and summary.recall_after == pytest.approx(0.25)
