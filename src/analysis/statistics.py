"""Simple statistics helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def degeneracy_flags(df: pd.DataFrame, label_name: str) -> Dict[str, object]:
    if df.empty or label_name not in df.columns:
        return {"degenerate": True, "reason": "missing data"}
    unique = df[label_name].dropna().unique()
    if len(unique) < 2:
        return {"degenerate": True, "reason": f"{label_name} has single value {unique.tolist()}"}
    return {"degenerate": False, "reason": ""}

