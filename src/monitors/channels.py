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
    "episode_length",
    "unique_positions",
    "revisit_ratio",
    "interact_fraction",
    "progress_delta",
    "path_efficiency",
    "action_hist_0",
    "action_hist_1",
    "action_hist_2",
    "action_hist_3",
    "action_hist_4",
]

RICH_FEATURES = RESTRICTED_FEATURES + [
    "final_distance_to_goal",
]

CHANNEL_SPECS: Dict[str, ChannelSpec] = {
    "restricted": ChannelSpec(
        name="restricted",
        feature_names=RESTRICTED_FEATURES,
        suspicious_features=["progress_delta", "path_efficiency"],
        description="Agent observations, actions, episode length, and compact progress features after leakage-driven removal of reward aggregates.",
    ),
    "rich": ChannelSpec(
        name="rich",
        feature_names=RICH_FEATURES,
        suspicious_features=["final_distance_to_goal"],
        description="Restricted features plus privileged but non-label-tautological environment-side summaries.",
    ),
}


def channel_matrix(feature_rows: List[dict], channel_name: str):
    spec = CHANNEL_SPECS[channel_name]
    rows = []
    for row in feature_rows:
        rows.append([float(row.get(name, 0.0)) for name in spec.feature_names])
    return spec, rows
