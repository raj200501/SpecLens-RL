"""Small metric utilities shared across analysis code."""

from __future__ import annotations

import math
from typing import Iterable


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def safe_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_vals = list(x)
    y_vals = list(y)
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return float("nan")
    mean_x = safe_mean(x_vals)
    mean_y = safe_mean(y_vals)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x_vals, y_vals))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x_vals))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y_vals))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)

