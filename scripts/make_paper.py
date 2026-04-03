#!/usr/bin/env python3
"""Populate paper placeholder sections from generated artifacts when available."""

from __future__ import annotations

from _bootstrap import ROOT
from src.analysis.paper_exports import write_experiment_section, write_results_section


def build_paper_assets() -> None:
    write_experiment_section(ROOT / "results", ROOT / "latex")
    write_results_section(ROOT / "results", ROOT / "latex")


def main() -> None:
    build_paper_assets()


if __name__ == "__main__":
    main()
