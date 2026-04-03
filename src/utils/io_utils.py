"""Filesystem helpers."""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(text)


def write_csv_gz(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    target = Path(path)
    ensure_dir(target.parent)
    fieldnames = list(rows[0].keys())
    with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_inventory(
    inventory_path: str | Path,
    artifact_path: str | Path,
    category: str,
    notes: str = "",
) -> None:
    inventory = Path(inventory_path)
    ensure_dir(inventory.parent)
    artifact = Path(artifact_path)
    size_bytes = artifact.stat().st_size if artifact.exists() else 0
    created_at = datetime.now(timezone.utc).isoformat()
    line = {
        "path": str(artifact),
        "size_bytes": size_bytes,
        "category": category,
        "created_at": created_at,
        "notes": notes,
    }
    exists = inventory.exists()
    with inventory.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(line.keys()))
        if not exists or inventory.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(line)

