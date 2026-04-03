"""Export helpers for schema-related paper tables."""

from __future__ import annotations

from typing import List

from .registry import get_task_schema


def environment_summary_rows(env_names: List[str]) -> List[dict]:
    rows = []
    for env_name in env_names:
        schema = get_task_schema(env_name)
        rows.append(
            {
                "env": schema.name,
                "proxy_reward_summary": schema.proxy_reward_name,
                "true_objective_summary": schema.true_objective_name,
                "corruption_channel_summary": "; ".join(schema.corruption_channels),
                "monitor_channel_summary": "; ".join(schema.monitor_channels),
                "ood_shift_summary": "; ".join(schema.shift_family),
            }
        )
    return rows

