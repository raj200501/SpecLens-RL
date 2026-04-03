"""Table generation."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.schema.export import environment_summary_rows
from src.utils.latex_utils import write_simple_table


def environment_summary_table(env_names: List[str], output_path: str | Path) -> None:
    rows = environment_summary_rows(env_names)
    write_simple_table(
        output_path,
        headers=["env", "proxy reward", "true objective", "corruption", "monitor", "OOD shift"],
        rows=[[row["env"], row["proxy_reward_summary"], row["true_objective_summary"], row["corruption_channel_summary"], row["monitor_channel_summary"], row["ood_shift_summary"]] for row in rows],
        caption="Environment summary for the SpecLens-RL benchmark.",
        label="tab:environment-summary",
    )


def dataframe_to_table(df: pd.DataFrame, output_path: str | Path, caption: str, label: str) -> None:
    if df.empty:
        return
    rows = []
    for row in df.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                rendered.append("--")
            elif isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(value)
        rows.append(rendered)
    write_simple_table(output_path, headers=list(df.columns), rows=rows, caption=caption, label=label)
