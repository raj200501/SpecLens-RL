import pandas as pd
import pytest

from src.analysis.estimators import (
    generalization_metrics,
    seed_dominance_summary,
    seed_variability_summary,
    specification_gap_metrics,
    threshold_robustness,
)


def test_specification_gap_metrics_smoke():
    df = pd.DataFrame(
        [
            {"proxy_return": 1.0, "true_score": 0.0, "reward_hack": 1, "method": "A"},
            {"proxy_return": 0.0, "true_score": 1.0, "reward_hack": 0, "method": "B"},
        ]
    )
    metrics = specification_gap_metrics(df)
    assert "proxy_true_correlation" in metrics
    assert metrics["hacking_incidence"] == 0.5


def test_generalization_metrics_smoke():
    id_df = pd.DataFrame([{"true_score": 1.0, "reward_hack": 0}])
    ood_df = pd.DataFrame([{"true_score": 0.5, "reward_hack": 1}])
    metrics = generalization_metrics(id_df, ood_df)
    assert metrics["true_score_ID"] == 1.0
    assert metrics["hacking_OOD"] == 1.0


def test_threshold_robustness_rows():
    df = pd.DataFrame([{"hack_score": 0.0}, {"hack_score": 2.0}, {"hack_score": 4.0}])
    rows = threshold_robustness(df, thresholds=[1.0, 3.0])
    assert rows[0]["hacking_incidence"] == 2 / 3
    assert rows[1]["hacking_incidence"] == 1 / 3


def test_seed_variability_summary():
    episodes = pd.DataFrame(
        [
            {"env_name": "ProxyFarming-v0", "method": "PROXY_PPO", "seed": 0, "split": "id", "true_score": -2.0, "reward_hack": 1, "proxy_return": 10.0},
            {"env_name": "ProxyFarming-v0", "method": "PROXY_PPO", "seed": 1, "split": "id", "true_score": -1.0, "reward_hack": 0, "proxy_return": 8.0},
            {"env_name": "ProxyFarming-v0", "method": "ORACLE_TRUE_PPO", "seed": 0, "split": "id", "true_score": 1.0, "reward_hack": 0, "proxy_return": 0.0},
            {"env_name": "ProxyFarming-v0", "method": "ORACLE_TRUE_PPO", "seed": 1, "split": "id", "true_score": 0.5, "reward_hack": 0, "proxy_return": 0.0},
        ]
    )
    summary = seed_variability_summary(episodes)
    proxy_row = summary[(summary["env"] == "ProxyFarming-v0") & (summary["method"] == "PROXY_PPO")].iloc[0]
    assert proxy_row["true_score_mean"] == -1.5
    assert proxy_row["hack_mean"] == 0.5
    assert proxy_row["true_score_std"] > 0.0


def test_seed_dominance_summary():
    episodes = pd.DataFrame(
        [
            {"env_name": "SensorTamper-v0", "method": "PROXY_PPO", "seed": 0, "split": "id", "true_score": -2.0, "reward_hack": 1, "proxy_return": 10.0},
            {"env_name": "SensorTamper-v0", "method": "PROXY_PPO", "seed": 1, "split": "id", "true_score": -1.0, "reward_hack": 1, "proxy_return": 11.0},
            {"env_name": "SensorTamper-v0", "method": "ORACLE_TRUE_PPO", "seed": 0, "split": "id", "true_score": 0.7, "reward_hack": 0, "proxy_return": 0.0},
            {"env_name": "SensorTamper-v0", "method": "ORACLE_TRUE_PPO", "seed": 1, "split": "id", "true_score": 0.9, "reward_hack": 0, "proxy_return": 0.0},
        ]
    )
    summary = seed_dominance_summary(episodes)
    row = summary.iloc[0]
    assert row["proxy_max_true_score"] == -1.0
    assert row["oracle_min_true_score"] == 0.7
    assert row["dominance_margin"] == pytest.approx(1.7)
    assert row["ordering_preserved"] == 1
