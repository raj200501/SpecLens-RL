"""Trajectory features for monitors, organised by what information they use.

A monitor channel is defined by the information it may use, not by a list of
hand-picked feature names. Every feature declares its *provenance*: the set of
trajectory fields it is computed from. A channel is the set of fields a monitor
is allowed to see, and it gets every feature whose provenance it covers.

* ``behavior``   -- positions and actions only; no knowledge of the task.
* ``task``       -- additionally knows the goal cell (the intended outcome).
* ``privileged`` -- additionally knows where the corruption channel can be
  accessed. Hack labels are defined from exactly this information, so this
  channel is leaky by construction and serves as a positive control for the
  leakage screen in :mod:`speclens.monitors.evaluate`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from speclens.envs.base import ACTION_NAMES, INTERACT

Traj = dict

CHANNELS: dict[str, frozenset[str]] = {
    "behavior": frozenset({"positions", "actions"}),
    "task": frozenset({"positions", "actions", "goal"}),
    "privileged": frozenset({"positions", "actions", "goal", "exploit_cells"}),
}
# Fields from which the hack labels are computed (see EpisodeStats.hack).
LABEL_SOURCES = frozenset({"positions", "actions", "exploit_cells"})


@dataclass(frozen=True)
class Feature:
    name: str
    sources: frozenset[str]
    fn: Callable[[Traj], float]


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _moves(traj: Traj) -> list[tuple[list[int], list[int], int]]:
    pos = traj["positions"]
    return list(zip(pos[:-1], pos[1:], traj["actions"]))


def _stays(traj: Traj) -> list[bool]:
    return [a == b for a, b, _ in _moves(traj)]


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
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}
    pairs = [(a, b) for a, b in pairwise(actions) if a in opposite and b in opposite]
    return sum(opposite[a] == b for a, b in pairs) / max(1, len(pairs))


def _action_fraction(action: int) -> Callable[[Traj], float]:
    return lambda tr: tr["actions"].count(action) / _n(tr)


def _goal_distances(traj: Traj) -> list[int]:
    return [_manhattan(p, traj["goal"]) for p in traj["positions"]]


def _on_exploit(traj: Traj) -> list[bool]:
    cells = {tuple(c) for c in traj["exploit_cells"]}
    return [tuple(p) in cells for p in traj["positions"][1:]]


def _n(traj: Traj) -> int:
    return max(1, len(traj["actions"]))


B = CHANNELS["behavior"]
T = CHANNELS["task"]
P = CHANNELS["privileged"]

FEATURES: tuple[Feature, ...] = (
    # behavior: positions and actions only
    Feature("length_ratio", B, lambda tr: len(tr["actions"]) / tr["max_steps"]),
    *(Feature(f"frac_{name}", B, _action_fraction(i)) for i, name in enumerate(ACTION_NAMES)),
    Feature("action_entropy", B, lambda tr: _action_entropy(tr["actions"])),
    Feature("repeat_action_frac", B, lambda tr: sum(a == b for a, b in pairwise(tr["actions"])) / _n(tr)),
    Feature("reversal_frac", B, lambda tr: _reversals(tr["actions"])),
    Feature("stay_frac", B, lambda tr: sum(_stays(tr)) / _n(tr)),
    Feature("longest_stay_ratio", B, lambda tr: _longest_run(_stays(tr)) / _n(tr)),
    Feature("unique_cell_ratio", B, lambda tr: len({tuple(p) for p in tr["positions"]}) / len(tr["positions"])),
    Feature("max_cell_share", B, lambda tr: max(Counter(map(tuple, tr["positions"])).values()) / len(tr["positions"])),
    Feature(
        "net_displacement",
        B,
        lambda tr: _manhattan(tr["positions"][0], tr["positions"][-1]) / (2 * (tr["size"] - 1)),
    ),
    Feature(
        "path_efficiency",
        B,
        lambda tr: _manhattan(tr["positions"][0], tr["positions"][-1]) / max(1, sum(not s for s in _stays(tr))),
    ),
    # task: also knows the goal cell
    Feature("final_goal_dist", T, lambda tr: _goal_distances(tr)[-1] / (2 * (tr["size"] - 1))),
    Feature("min_goal_dist", T, lambda tr: min(_goal_distances(tr)) / (2 * (tr["size"] - 1))),
    Feature(
        "goal_progress",
        T,
        lambda tr: (_goal_distances(tr)[0] - _goal_distances(tr)[-1]) / max(1, _goal_distances(tr)[0]),
    ),
    Feature(
        "frac_steps_closer",
        T,
        lambda tr: sum(b < a for a, b in pairwise(_goal_distances(tr))) / _n(tr),
    ),
    Feature(
        "frac_steps_farther",
        T,
        lambda tr: sum(b > a for a, b in pairwise(_goal_distances(tr))) / _n(tr),
    ),
    # privileged: also knows where the corruption channel is
    Feature("frac_on_exploit", P, lambda tr: sum(_on_exploit(tr)) / _n(tr)),
    Feature(
        "interact_on_exploit",
        P,
        lambda tr: sum(on and a == INTERACT for on, a in zip(_on_exploit(tr), tr["actions"])) / _n(tr),
    ),
    Feature(
        "exploit_visits",
        P,
        lambda tr: sum(1 for prev, cur in zip([False, *_on_exploit(tr)], _on_exploit(tr)) if cur and not prev) / _n(tr),
    ),
)


def channel_features(channel: str) -> list[str]:
    allowed = CHANNELS[channel]
    return [f.name for f in FEATURES if f.sources <= allowed]


def uses_label_sources(channel: str) -> bool:
    """True if the channel can see every field the hack label is computed from."""
    return CHANNELS[channel] >= LABEL_SOURCES


def extract_features(traj: Traj) -> dict[str, float]:
    return {f.name: float(f.fn(traj)) for f in FEATURES}
