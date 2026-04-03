from src.monitors.channels import CHANNEL_SPECS, channel_matrix


def test_monitor_channels_exclude_direct_labels():
    for spec in CHANNEL_SPECS.values():
        banned = {"reward_hack", "true_score", "success", "tamper_count"}
        assert banned.isdisjoint(set(spec.feature_names))


def test_channel_matrix_shapes():
    rows = [{"episode_length": 1.0, "reward_mean": 0.0}]
    spec, matrix = channel_matrix(rows, "restricted")
    assert len(matrix) == 1
    assert len(matrix[0]) == len(spec.feature_names)

