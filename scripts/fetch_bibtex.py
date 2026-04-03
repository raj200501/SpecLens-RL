#!/usr/bin/env python3
"""Preserve seed bibliography entries and flag unresolved metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    path = Path(ROOT) / "latex" / "references_seed.bib"
    text = path.read_text(encoding="utf-8")
    unresolved = "TODO verify exact citation metadata" in text or "metadata should be rechecked" in text
    if args.check_only:
        print({"path": str(path), "has_unresolved_entries": unresolved})
        return
    print(
        "Automatic remote bibliography fetch is not implemented in the offline-safe core pipeline. "
        "Existing seed entries were preserved and unresolved metadata remains explicitly marked."
    )


if __name__ == "__main__":
    main()
