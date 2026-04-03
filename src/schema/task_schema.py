"""Structured task schema objects for specification-diagnostic RL tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class TaskSchema:
    name: str
    state_description: str
    action_description: str
    observation_description: str
    proxy_reward_name: str
    true_objective_name: str
    corruption_channels: List[str]
    monitor_channels: List[str]
    shift_family: List[str]
    success_definition: str
    reward_hack_definition: str
    notes: str = ""
    compact_description: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

