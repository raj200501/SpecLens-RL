import numpy as np
import pandas as pd
import pytest

from speclens.analysis.stats import bootstrap_ci, summarize
from speclens.report import dominance


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(1.0, 0.5, size=10)
    lo, hi = bootstrap_ci(values)
    assert lo < values.mean() < hi
    assert np.isnan(bootstrap_ci([1.0])[0])


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
