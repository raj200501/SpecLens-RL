"""Trajectory feature extraction for monitor channels."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from src.envs.common import manhattan


def _positions(trajectory: List[dict]) -> List[tuple[int, int]]:
    return [tuple(step["agent_pos"]) for step in trajectory]


def _risk_visit_count(trajectory: List[dict], risky_positions: List[List[int]]) -> int:
    risky = {tuple(pos) for pos in risky_positions}
    return sum(1 for step in trajectory if tuple(step["agent_pos"]) in risky)


def extract_episode_features(episode_record: Dict[str, object]) -> Dict[str, float]:
    trajectory = episode_record["trajectory"]
    summary = episode_record["summary"]
    context = episode_record["context"]
    positions = _positions(trajectory)
    rewards = np.array([step["proxy_reward"] for step in trajectory], dtype=np.float32)
    actions = [step["action"] for step in trajectory]
    action_hist = Counter(actions)
    unique_positions = len(set(positions))
    revisit_ratio = 0.0 if not positions else 1.0 - unique_positions / len(positions)
    risky_visits = _risk_visit_count(trajectory, context.get("risky_positions", []))
    goal_pos = tuple(context.get("goal_pos", [0, 0]))
    final_distance = manhattan(positions[-1], goal_pos) if positions else 0
    start_distance = manhattan(positions[0], goal_pos) if positions else 0
    progress = float(start_distance - final_distance)
    interact_fraction = action_hist.get(4, 0) / max(1, len(actions))
    features = {
        "episode_length": float(summary.get("episode_length", len(trajectory))),
        "proxy_return": float(summary.get("proxy_return", float(rewards.sum()) if len(rewards) else 0.0)),
        "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
        "reward_std": float(rewards.std()) if len(rewards) else 0.0,
        "positive_reward_steps": float((rewards > 0).sum()),
        "unique_positions": float(unique_positions),
        "revisit_ratio": float(revisit_ratio),
        "interact_fraction": float(interact_fraction),
        "risky_visit_count": float(risky_visits),
        "risky_visit_fraction": float(risky_visits / max(1, len(trajectory))),
        "progress_delta": float(progress),
        "final_distance_to_goal": float(final_distance),
        "path_efficiency": float(progress / max(1, len(trajectory))),
        "split_is_ood": 1.0 if episode_record["split"] == "ood" else 0.0,
    }
    for action in range(5):
        features[f"action_hist_{action}"] = float(action_hist.get(action, 0) / max(1, len(actions)))
    env_name = episode_record["env_name"]
    if env_name == "ProxyFarming-v0":
        features["beacon_pressure"] = float(summary.get("beacon_interaction_count", 0) / max(1, summary.get("episode_length", 1)))
    elif env_name == "SensorTamper-v0":
        features["sensor_activity"] = float(summary.get("sensor_value", 0.0))
        features["tamper_presence"] = 1.0 if risky_visits > 0 else 0.0
    elif env_name == "GemTimeout-v0":
        features["gem_pressure"] = float(summary.get("gems_collected", 0) / max(1, summary.get("episode_length", 1)))
    return features

