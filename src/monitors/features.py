"""Trajectory feature extraction for monitor channels."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from src.envs.common import manhattan


OPPOSITE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}


def _positions(trajectory: List[dict]) -> List[tuple[int, int]]:
    return [tuple(step["agent_pos"]) for step in trajectory]


def _longest_still_streak(positions: List[tuple[int, int]]) -> int:
    if not positions:
        return 0
    best = 1
    current = 1
    for prev, cur in zip(positions, positions[1:]):
        if cur == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _repeat_action_fraction(actions: List[int]) -> float:
    if len(actions) < 2:
        return 0.0
    repeats = sum(1 for prev, cur in zip(actions, actions[1:]) if prev == cur)
    return repeats / (len(actions) - 1)


def _reversal_fraction(actions: List[int]) -> float:
    if len(actions) < 2:
        return 0.0
    reversals = 0
    total = 0
    for prev, cur in zip(actions, actions[1:]):
        if prev == 4 or cur == 4:
            continue
        total += 1
        if OPPOSITE_ACTION.get(prev) == cur:
            reversals += 1
    return reversals / max(1, total)


def _turn_fraction(actions: List[int]) -> float:
    if len(actions) < 2:
        return 0.0
    turns = 0
    total = 0
    for prev, cur in zip(actions, actions[1:]):
        if prev == 4 or cur == 4:
            continue
        total += 1
        if prev != cur:
            turns += 1
    return turns / max(1, total)


def _normalized_action_entropy(actions: List[int]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    probs = np.array([count / len(actions) for count in counts.values()], dtype=np.float32)
    entropy = float(-(probs * np.log(probs + 1e-8)).sum())
    return entropy / np.log(5.0)


def extract_episode_features(episode_record: Dict[str, object]) -> Dict[str, float]:
    trajectory = episode_record["trajectory"]
    context = episode_record["context"]
    positions = _positions(trajectory)
    actions = [int(step["action"]) for step in trajectory]
    goal_pos = tuple(context.get("goal_pos", positions[-1] if positions else (0, 0)))
    start_pos = tuple(context.get("start_pos", positions[0] if positions else (0, 0)))
    grid_size = int(context.get("grid_size", 1))
    max_steps = int(context.get("max_steps", max(1, len(trajectory))))

    episode_length = len(trajectory)
    unique_positions = len(set(positions))
    unique_position_ratio = unique_positions / max(1, episode_length)
    revisit_ratio = 1.0 - unique_position_ratio
    interact_fraction = sum(1 for action in actions if action == 4) / max(1, episode_length)
    movement_fraction = 1.0 - interact_fraction

    start_distance = manhattan(start_pos, goal_pos)
    distances = [manhattan(pos, goal_pos) for pos in positions]
    final_distance = distances[-1] if distances else start_distance
    min_distance = min(distances) if distances else start_distance
    distance_scale = max(1.0, float(start_distance))

    first_third = distances[: max(1, len(distances) // 3)] if distances else [start_distance]
    last_third = distances[-max(1, len(distances) // 3) :] if distances else [start_distance]
    early_mean = float(np.mean(first_third))
    late_mean = float(np.mean(last_third))

    improvement_flags = []
    previous_distance = start_distance
    for distance in distances:
        improvement_flags.append(1.0 if distance < previous_distance else 0.0)
        previous_distance = distance

    if positions:
        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    else:
        bbox_area = 1

    start_return_fraction = 0.0
    if positions:
        offset = len(positions) // 4
        late_positions = positions[offset:] if offset < len(positions) else positions
        start_return_fraction = sum(1 for pos in late_positions if manhattan(pos, start_pos) <= 1) / max(1, len(late_positions))

    goal_band_fraction = sum(1 for distance in distances if distance <= 2) / max(1, len(distances))
    late_goal_band_fraction = sum(1 for distance in last_third if distance <= 2) / max(1, len(last_third))

    features = {
        "episode_length_ratio": float(episode_length / max(1, max_steps)),
        "movement_fraction": float(movement_fraction),
        "interact_fraction": float(interact_fraction),
        "repeat_action_fraction": float(_repeat_action_fraction(actions)),
        "reversal_fraction": float(_reversal_fraction(actions)),
        "turn_fraction": float(_turn_fraction(actions)),
        "action_entropy": float(_normalized_action_entropy(actions)),
        "unique_position_ratio": float(unique_position_ratio),
        "revisit_ratio": float(revisit_ratio),
        "bbox_area_ratio": float(bbox_area / max(1, grid_size * grid_size)),
        "distance_progress_ratio": float((start_distance - final_distance) / distance_scale),
        "late_progress_ratio": float((early_mean - late_mean) / distance_scale),
        "distance_improvement_fraction": float(sum(improvement_flags) / max(1, len(improvement_flags))),
        "still_streak_ratio": float(_longest_still_streak(positions) / max(1, episode_length)),
        "goal_band_fraction": float(goal_band_fraction),
        "late_goal_band_fraction": float(late_goal_band_fraction),
        "start_return_fraction": float(start_return_fraction),
        "min_distance_ratio": float(min_distance / distance_scale),
        "final_distance_ratio": float(final_distance / distance_scale),
        "distance_volatility": float(np.std(distances) / distance_scale) if distances else 0.0,
    }
    return features
