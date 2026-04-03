"""Monitor dataset building from saved evaluation trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .features import extract_episode_features


def load_episode_records(run_dir: str | Path) -> List[Dict[str, object]]:
    path = Path(run_dir) / "trajectories.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_monitor_dataset(results_root: str | Path, env_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in Path(results_root).glob(f"runs/{env_name}/*/seed_*"):
        for record in load_episode_records(path):
            feature_row = extract_episode_features(record)
            feature_row.update(
                {
                    "env_name": record["env_name"],
                    "method": record["method"],
                    "seed": int(record.get("seed", -1)),
                    "run_id": f"{record['env_name']}::{record['method']}::seed_{int(record.get('seed', -1))}",
                    "split": record["split"],
                    "label_hack": int(record["summary"].get("reward_hack", False)),
                    "label_failure": int(not record["summary"].get("success", False)),
                    "source_tamper_used": int(record["summary"].get("tamper_used", False)),
                    "source_beacon_active": int(record["summary"].get("beacon_interaction_count", 0) > 0),
                    "source_gem_heavy": int(record["summary"].get("gems_collected", 0) >= 2),
                }
            )
            rows.append(feature_row)
    return rows
