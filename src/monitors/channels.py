"""Explicit monitor-channel feature specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    feature_names: List[str]
    suspicious_features: List[str]
    description: str


RESTRICTED_FEATURES = [
    "episode_length_ratio",
    "movement_fraction",
    "interact_fraction",
    "repeat_action_fraction",
    "reversal_fraction",
    "turn_fraction",
    "action_entropy",
    "unique_position_ratio",
    "revisit_ratio",
    "bbox_area_ratio",
    "distance_progress_ratio",
    "late_progress_ratio",
    "distance_improvement_fraction",
    "still_streak_ratio",
]

RICH_FEATURES = RESTRICTED_FEATURES + [
    "goal_band_fraction",
    "late_goal_band_fraction",
    "start_return_fraction",
    "min_distance_ratio",
    "final_distance_ratio",
    "distance_volatility",
]

CHANNEL_SPECS: Dict[str, ChannelSpec] = {
    "restricted": ChannelSpec(
        name="restricted",
        feature_names=RESTRICTED_FEATURES,
        suspicious_features=[],
        description="Behavior-level audit features built from action patterns, revisitation, coverage, and coarse progress trends.",
    ),
    "rich": ChannelSpec(
        name="rich",
        feature_names=RICH_FEATURES,
        suspicious_features=[],
        description="Restricted features plus richer distance-profile summaries that remain trajectory-level rather than event-counter based.",
    ),
}


def channel_matrix(feature_rows: List[dict], channel_name: str):
    spec = CHANNEL_SPECS[channel_name]
    rows = []
    for row in feature_rows:
        rows.append([float(row.get(name, 0.0)) for name in spec.feature_names])
    return spec, rows
