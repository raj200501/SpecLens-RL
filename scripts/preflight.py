#!/usr/bin/env python3
"""Collect local system information and choose a safe execution profile."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import ROOT

from src.utils.io_utils import write_json, write_text
from src.utils.system import collect_preflight, render_preflight_markdown


def main() -> None:
    results_dir = Path(ROOT) / "results"
    preflight = collect_preflight(ROOT)
    write_json(results_dir / "preflight.json", preflight)
    write_text(results_dir / "PREFLIGHT.md", render_preflight_markdown(preflight))
    print(preflight["selected_profile"])


if __name__ == "__main__":
    main()
