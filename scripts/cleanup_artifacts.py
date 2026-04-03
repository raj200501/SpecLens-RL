#!/usr/bin/env python3
"""Prune generated artifacts while preserving required outputs."""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    results_root = ROOT / "results" / "runs"
    removed = []
    for run_dir in results_root.glob("*/*/seed_*"):
        for path in run_dir.iterdir():
            if path.name in {"latest.ckpt", "best.ckpt", "summary.json", "eval_summary.json", "episodes.csv.gz", "trajectories.json", "train_metrics.csv.gz"}:
                continue
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    print(json.dumps({"removed_count": len(removed)}))


if __name__ == "__main__":
    main()
