from src.monitors.leakage import run_leakage_checks


def test_leakage_warns_when_behavior_predicts_label_source():
    rows = []
    for idx in range(30):
        label = int(idx >= 10)
        rows.append(
            {
                "episode_length_ratio": 0.5 + 0.4 * label,
                "movement_fraction": 0.9 - 0.2 * label,
                "interact_fraction": 0.1 + 0.2 * label,
                "repeat_action_fraction": 0.1 * label,
                "reversal_fraction": 0.0,
                "turn_fraction": 0.2,
                "action_entropy": 0.4,
                "unique_position_ratio": 0.9 - 0.4 * label,
                "revisit_ratio": 0.1 + 0.4 * label,
                "bbox_area_ratio": 0.2,
                "distance_progress_ratio": 0.8 - 0.5 * label,
                "late_progress_ratio": 0.7 - 0.6 * label,
                "distance_improvement_fraction": 0.8 - 0.5 * label,
                "still_streak_ratio": 0.1 + 0.5 * label,
                "goal_band_fraction": 0.8 - 0.7 * label,
                "late_goal_band_fraction": 0.8 - 0.7 * label,
                "start_return_fraction": 0.0 + 0.7 * label,
                "min_distance_ratio": 0.1 + 0.6 * label,
                "final_distance_ratio": 0.1 + 0.7 * label,
                "distance_volatility": 0.1 + 0.2 * label,
                "label_hack": label,
                "source_tamper_used": label,
                "source_beacon_active": 0,
                "source_gem_heavy": 0,
            }
        )
    result = run_leakage_checks(rows, "ProxyFarming-v0", "rich", "label_hack")
    assert result.status == "warning"
    assert result.max_behavioral_auc is not None and result.max_behavioral_auc > 0.99
