#!/usr/bin/env python3
"""Populate paper placeholder sections from generated artifacts when available."""

from __future__ import annotations

import argparse

from _bootstrap import ROOT
from src.analysis.paper_exports import write_experiment_section, write_results_section


def build_paper_assets(results_root=None, latex_root=None) -> None:
    results_root = results_root or (ROOT / "results")
    latex_root = latex_root or (ROOT / "latex")
    write_experiment_section(results_root, latex_root)
    write_results_section(results_root, latex_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root")
    parser.add_argument("--latex-root")
    args = parser.parse_args()
    build_paper_assets(
        results_root=(ROOT / args.results_root) if args.results_root else None,
        latex_root=(ROOT / args.latex_root) if args.latex_root else None,
    )


if __name__ == "__main__":
    main()
