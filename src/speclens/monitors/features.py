"""Trajectory features for monitors, organised by what information they use.

A monitor channel is defined by the information it may use, not by a list of
hand-picked feature names. Every feature declares its *provenance*: the set of
trajectory fields it is computed from. A channel is the set of fields a monitor
is allowed to see, and it gets every feature whose provenance it covers.
Provenance is enforced, not just declared: each feature is computed on a copy of
the trajectory that contains only its declared fields (plus the public grid
size and step limit), so a feature that reads anything else raises ``KeyError``.

* ``behavior``   -- positions and actions. This includes the episode's length,
  so the monitor can tell whether an episode ended before the step limit; in
  tasks whose episodes end on success, that is itself informative.
* ``task``       -- additionally the goal cell (the intended outcome).
* ``privileged`` -- additionally the cells through which the corruption channel
  is accessed. Hack labels are computed from these fields, so this channel can
  reconstruct the label exactly and serves as a positive control for the
  admissibility screen in :mod:`speclens.analysis.monitoring`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TypedDict

import numpy as np

from speclens.envs import ENVS
from speclens.envs.base import ACTION_NAMES, DOWN, INTERACT, LEFT, RIGHT, UP


class Trajectory(TypedDict):
    """What a :class:`speclens.envs.GridTask` exposes about one finished episode."""

    positions: list[list[int]]  # agent cell before the first step and after every step
    actions: list[int]
    goal: list[int]
    start: list[int]
    walls: list[list[int]]
    exploit_cells: list[list[int]]
    size: int
    max_steps: int


PUBLIC_FIELDS = frozenset({"size", "max_steps"})  # constants of the task, known to every monitor
BEHAVIOR = frozenset({"positions", "actions"})
TASK = BEHAVIOR | {"goal"}
PRIVILEGED = TASK | {"exploit_cells"}
CHANNELS: dict[str, frozenset[str]] = {"behavior": BEHAVIOR, "task": TASK, "privileged": PRIVILEGED}


@dataclass(frozen=True)
class Feature:
    name: str
    sources: frozenset[str]
    fn: Callable[[dict], float]


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _n(traj: dict) -> int:
    return max(1, len(traj["actions"]))


def _stays(traj: dict) -> list[bool]:
    return [a == b for a, b in pairwise(traj["positions"])]


def _longest_run(flags: list[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _action_entropy(actions: list[int]) -> float:
    if not actions:
        return 0.0
    p = np.array(list(Counter(actions).values()), dtype=float) / len(actions)
    return float(-(p * np.log(p)).sum() / np.log(len(ACTION_NAMES)))


def _reversals(actions: list[int]) -> float:
    opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
    pairs = [(a, b) for a, b in pairwise(actions) if a in opposite and b in opposite]
    return sum(opposite[a] == b for a, b in pairs) / max(1, len(pairs))


def _action_fraction(action: int) -> Callable[[dict], float]:
    return lambda tr: tr["actions"].count(action) / _n(tr)


def _goal_distances(traj: dict) -> list[int]:
    return [_manhattan(p, traj["goal"]) for p in traj["positions"]]


def _on_exploit(traj: dict) -> list[bool]:
    cells = {tuple(c) for c in traj["exploit_cells"]}
    return [tuple(p) in cells for p in traj["positions"][1:]]


def _diameter(traj: dict) -> int:
    return 2 * (traj["size"] - 1)


FEATURES: tuple[Feature, ...] = (
    # behavior
    Feature("length_ratio", BEHAVIOR, lambda tr: len(tr["actions"]) / tr["max_steps"]),
    *(Feature(f"frac_{name}", BEHAVIOR, _action_fraction(i)) for i, name in enumerate(ACTION_NAMES)),
    Feature("action_entropy", BEHAVIOR, lambda tr: _action_entropy(tr["actions"])),
    Feature("repeat_action_frac", BEHAVIOR, lambda tr: sum(a == b for a, b in pairwise(tr["actions"])) / _n(tr)),
    Feature("reversal_frac", BEHAVIOR, lambda tr: _reversals(tr["actions"])),
    Feature("stay_frac", BEHAVIOR, lambda tr: sum(_stays(tr)) / _n(tr)),
    Feature("longest_stay_ratio", BEHAVIOR, lambda tr: _longest_run(_stays(tr)) / _n(tr)),
    Feature("unique_cell_ratio", BEHAVIOR, lambda tr: len({tuple(p) for p in tr["positions"]}) / len(tr["positions"])),
    Feature(
        "max_cell_share", BEHAVIOR, lambda tr: max(Counter(map(tuple, tr["positions"])).values()) / len(tr["positions"])
    ),
    Feature(
        "net_displacement", BEHAVIOR, lambda tr: _manhattan(tr["positions"][0], tr["positions"][-1]) / _diameter(tr)
    ),
    Feature(
        "path_efficiency",
        BEHAVIOR,
        lambda tr: _manhattan(tr["positions"][0], tr["positions"][-1]) / max(1, sum(not s for s in _stays(tr))),
    ),
    # task: also knows the goal cell
    Feature("final_goal_dist", TASK, lambda tr: _goal_distances(tr)[-1] / _diameter(tr)),
    Feature("min_goal_dist", TASK, lambda tr: min(_goal_distances(tr)) / _diameter(tr)),
    Feature(
        "goal_progress",
        TASK,
        lambda tr: (_goal_distances(tr)[0] - _goal_distances(tr)[-1]) / max(1, _goal_distances(tr)[0]),
    ),
    Feature("frac_steps_closer", TASK, lambda tr: sum(b < a for a, b in pairwise(_goal_distances(tr))) / _n(tr)),
    Feature("frac_steps_farther", TASK, lambda tr: sum(b > a for a, b in pairwise(_goal_distances(tr))) / _n(tr)),
    # privileged: also knows where the corruption channel is
    Feature("frac_on_exploit", PRIVILEGED, lambda tr: sum(_on_exploit(tr)) / _n(tr)),
    Feature(
        "interact_on_exploit",
        PRIVILEGED,
        lambda tr: sum(on and a == INTERACT for on, a in zip(_on_exploit(tr), tr["actions"])) / _n(tr),
    ),
    Feature(
        "exploit_visits",
        PRIVILEGED,
        lambda tr: sum(cur and not prev for prev, cur in pairwise([False, *_on_exploit(tr)])) / _n(tr),
    ),
)


def channel_features(channel: str) -> list[str]:
    allowed = CHANNELS[channel]
    return [f.name for f in FEATURES if f.sources <= allowed]


def sees_label_sources(channel: str, env_name: str) -> bool:
    """True if the channel sees every field the task's hack label is computed from.

    Such a channel can reconstruct the label exactly, so a high monitor score on
    it says nothing about monitorability.
    """
    return ENVS[env_name].label_sources <= CHANNELS[channel]


def restrict(traj: Trajectory | dict, fields: frozenset[str]) -> dict:
    """The part of a trajectory a feature with the given provenance may read."""
    return {key: traj[key] for key in fields | PUBLIC_FIELDS}


def extract_features(traj: Trajectory | dict) -> dict[str, float]:
    return {f.name: float(f.fn(restrict(traj, f.sources))) for f in FEATURES}
