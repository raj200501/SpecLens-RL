import numpy as np
import pandas as pd
import pytest

from speclens.analysis.penalty import baseline_recall, evasion, monitor_scores, penalized_recall, summarize_penalty
from speclens.analysis.stats import mean_ci, summarize
from speclens.monitors.features import FEATURES
from speclens.monitors.penalty import LinearMonitor
from speclens.report import README_BEGIN, README_END, dominance, emergence, transition_steps, update_readme


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


def test_emergence_onset_and_success_before():
    curves = pd.DataFrame(
        {
            "env": "E",
            "seed": [0] * 4 + [1] * 4,
            "env_steps": [1, 2, 3, 4] * 2,
            "hack": [0.0, 0.2, 0.6, 0.9, 0.1, 0.1, 0.2, 0.4],
            "success": [0.5, 0.9, 0.3, 0.1, 0.2, 0.8, 0.7, 0.6],
        }
    )
    runs = emergence(curves).set_index("seed")
    assert runs.loc[0, "onset"] == 3 and runs.loc[0, "peak_success_before"] == pytest.approx(0.9)
    assert np.isnan(runs.loc[1, "onset"]) and runs.loc[1, "peak_success_before"] == pytest.approx(0.8)
    # Hack rate 0.0, 0.2, 0.6, 0.9: last snapshot below 10% is step 1, first above 75% is step 4.
    assert runs.loc[0, "transition"] == 3 and np.isnan(runs.loc[1, "transition"])
    assert transition_steps(curves[curves.seed == 0].assign(hack=[0.2, 0.3, 0.8, 0.9])) == 2  # never below 10%


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
    recall = penalized_recall(episodes, min_hacks=1).set_index("seed").recall
    assert recall.to_dict() == {0: 0.5, 1: 0.0}
    assert penalized_recall(episodes, min_hacks=2).seed.tolist() == [0]  # seed 1 has a single hack

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

    summary = summarize_penalty(episodes, pd.DataFrame(rows), "behavior", min_hacks=1).iloc[0]
    assert summary.undetected_hack == pytest.approx((1 / 3 + 1 / 3) / 2)
    assert summary.recall_before == 1.0 and summary.recall_after == pytest.approx(0.25)
    assert np.isnan(summarize_penalty(episodes, pd.DataFrame(rows), "behavior").iloc[0].recall_after)


def test_evasion_decomposes_the_change_in_monitor_logit_exactly():
    rng = np.random.default_rng(0)
    names = ("a", "b", "c")
    monitor = LinearMonitor(names, (0.5, 1.0, -2.0), (2.0, 0.5, 1.0), (1.5, -0.7, 0.3), -0.2)
    before = pd.DataFrame(rng.normal(size=(50, 3)), columns=names)
    after = pd.DataFrame(rng.normal(loc=[-1.0, 0.5, 0.0], size=(40, 3)), columns=names)
    shift = evasion(before, after, monitor)

    def mean_logit(data):
        p = monitor_scores(data, monitor)
        return np.log(p / (1 - p)).mean()

    assert shift.contribution.sum() == pytest.approx(mean_logit(after) - mean_logit(before))
    assert monitor_scores(before.iloc[:1], monitor)[0] == pytest.approx(monitor.score(before.iloc[0].to_dict()))


def test_update_readme_replaces_only_the_marked_section(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_BEGIN}\nold table\n{README_END}\noutro\n", encoding="utf-8")
    update_readme(readme, "new table")
    assert readme.read_text(encoding="utf-8") == f"intro\n{README_BEGIN}\nnew table\n{README_END}\noutro\n"
    readme.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(readme, "new table")
