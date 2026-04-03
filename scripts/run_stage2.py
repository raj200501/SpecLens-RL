#!/usr/bin/env python3
"""Optional Stage 2 extensions."""

from __future__ import annotations

import json

from _bootstrap import ROOT
from src.utils.io_utils import read_json, write_json


def main() -> None:
    preflight = read_json(ROOT / "results" / "preflight.json")
    profile = preflight["selected_profile"]
    summary = {
        "stage": "stage2",
        "status": "skipped",
        "profile": profile,
        "reason": "Stage 2 is intentionally not launched by default; enable only after Stage 1 artifacts are reviewed.",
    }
    write_json(ROOT / "results" / "stage2_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
