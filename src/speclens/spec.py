"""The specification-diagnostic task tuple.

A SpecLens-RL task is described by ``T = (M, R_p, J_t, C, M_mon, Delta, Pi)``:
dynamics, proxy reward, true objective, corruption channel, monitor view, shift
family and policy family. ``TaskSpec`` records the environment-side components
in plain language so they can be exported next to results; the policy family is
fixed by the experiment config rather than by the task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dynamics: str  # M
    proxy_reward: str  # R_p
    true_objective: str  # J_t
    corruption_channel: str  # C
    monitor_view: str  # M_mon
    shift: str  # Delta
    success: str
    hack: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
