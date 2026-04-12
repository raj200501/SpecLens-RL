#!/usr/bin/env python3
"""Flatten the modular NeurIPS paper into final submission artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from _bootstrap import ROOT
from src.utils.io_utils import write_text


LATEX_DIR = ROOT / "neurips_latex"
MAIN_TEX = LATEX_DIR / "main.tex"
FINAL_TEX = LATEX_DIR / "final.tex"
SNAPSHOT = LATEX_DIR / "final_flattened_source_snapshot.txt"


def expand_inputs(text: str, sources: list[Path]) -> str:
    pattern = re.compile(r"\\input\{([^}]+)\}")

    def repl(match: re.Match[str]) -> str:
        rel = match.group(1)
        input_path = (LATEX_DIR / rel).with_suffix(".tex")
        sources.append(input_path)
        nested = input_path.read_text(encoding="utf-8")
        return expand_inputs(nested, sources).strip()

    return pattern.sub(repl, text)


def ordered_unique(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def main() -> None:
    sources = [MAIN_TEX]
    final_text = expand_inputs(MAIN_TEX.read_text(encoding="utf-8"), sources)
    write_text(FINAL_TEX, final_text)

    ordered_sources = ordered_unique(sources)
    snapshot_lines = ["The flattened `final.tex` was assembled from:"]
    for path in ordered_sources:
        rel = path.relative_to(LATEX_DIR)
        snapshot_lines.append(f"- [{rel}]({LATEX_DIR.name}/{rel})")
    write_text(SNAPSHOT, "\n".join(snapshot_lines) + "\n")


if __name__ == "__main__":
    main()
