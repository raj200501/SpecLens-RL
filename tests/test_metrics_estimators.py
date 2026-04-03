import pandas as pd

from src.analysis.estimators import generalization_metrics, specification_gap_metrics, threshold_robustness


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

