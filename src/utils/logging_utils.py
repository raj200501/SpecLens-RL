"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import ensure_dir


def configure_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        target = Path(log_path)
        ensure_dir(target.parent)
        handlers.append(logging.FileHandler(target, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

