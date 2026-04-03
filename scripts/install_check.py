#!/usr/bin/env python3
"""Local import check without fabricating installation success."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import ROOT

from src.utils.io_utils import write_json, write_text
from src.utils.system import optional_import_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-requirements-report", action="store_true")
    args = parser.parse_args()

    report = {
        "numpy": optional_import_version("numpy"),
        "pandas": optional_import_version("pandas"),
        "matplotlib": optional_import_version("matplotlib"),
        "yaml": optional_import_version("yaml"),
        "sklearn": optional_import_version("sklearn"),
        "torch": optional_import_version("torch"),
        "gymnasium": optional_import_version("gymnasium"),
        "minigrid": optional_import_version("minigrid"),
    }

    if args.write_requirements_report:
        results_dir = Path(ROOT) / "results"
        write_json(results_dir / "install_check.json", report)
        lines = ["# Install Check", ""]
        for key, value in report.items():
            lines.append(f"- {key}: `{value}`")
        write_text(results_dir / "INSTALL_CHECK.md", "\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
