from src.monitors.leakage import run_leakage_checks


def test_leakage_detects_trivial_signal():
    rows = []
    for idx in range(20):
        label = int(idx >= 10)
        rows.append(
            {
                "episode_length": float(idx),
                "unique_positions": 1.0,
                "revisit_ratio": 0.0,
                "interact_fraction": 0.0,
                "progress_delta": 0.0,
                "path_efficiency": 0.0,
                "split_is_ood": 0.0,
                "action_hist_0": 1.0,
                "action_hist_1": 0.0,
                "action_hist_2": 0.0,
                "action_hist_3": 0.0,
                "action_hist_4": 0.0,
                "final_distance_to_goal": float(label),
                "label_hack": label,
                "source_tamper_used": label,
                "source_beacon_active": 0,
                "source_gem_heavy": 0,
            }
        )
    result = run_leakage_checks(rows, "ProxyFarming-v0", "rich", "label_hack")
    assert result.status == "fail"
