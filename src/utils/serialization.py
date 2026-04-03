"""Compact serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir


def save_npz(path: str | Path, **arrays: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    np.savez_compressed(target, **arrays)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    import torch

    target = Path(path)
    ensure_dir(target.parent)
    torch.save(payload, target)


def load_checkpoint(path: str | Path) -> dict:
    import torch

    return torch.load(Path(path), map_location="cpu")
