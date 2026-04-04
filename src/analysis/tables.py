"""Table generation."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.schema.export import environment_summary_rows
from src.utils.io_utils import write_text
from src.utils.latex_utils import write_simple_table


METHOD_LABELS = {
    "PROXY_PPO": "Proxy PPO",
    "ORACLE_TRUE_PPO": "Oracle True PPO",
}

ENV_LABELS = {
    "ProxyFarming-v0": "ProxyFarming",
    "SensorTamper-v0": "SensorTamper",
    "GemTimeout-v0": "GemTimeout",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}


def _write_custom_table(
    output_path: str | Path,
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    label: str,
    column_spec: str,
    table_env: str = "table",
    size_command: str = r"\small",
    midrules_after: List[int] | None = None,
) -> None:
    target = Path(output_path)
    midrules_after = set(midrules_after or [])
    lines = [
        rf"\begin{{{table_env}}}[t]",
        r"\centering",
        size_command,
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        lines.append(" & ".join(row) + r" \\")
        if idx in midrules_after:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\end{{{table_env}}}",
            "",
        ]
    )
    write_text(target, "\n".join(lines))


def environment_summary_table(env_names: List[str], output_path: str | Path) -> None:
    rows = environment_summary_rows(env_names)
    write_simple_table(
        output_path,
        headers=["env", "proxy reward", "true objective", "corruption", "monitor", "OOD shift"],
        rows=[[row["env"], row["proxy_reward_summary"], row["true_objective_summary"], row["corruption_channel_summary"], row["monitor_channel_summary"], row["ood_shift_summary"]] for row in rows],
        caption="Environment summary for the SpecLens-RL benchmark.",
        label="tab:environment-summary",
    )


def environment_main_table(output_path: str | Path) -> None:
    rows = [
        ["ProxyFarming", "Beacon reward", "Reach goal efficiently", "Repeated beacon harvesting", "Beacon move; layout; distractors"],
        ["SensorTamper", "Sensor-mediated reward", "Deliver object to target", "Tamper tile spoofs sensor", "Tamper move; delay; path change"],
        ["GemTimeout", "Gem reward", "Reach goal before timeout", "Gem detours trade off with time", "Density; budget; deceptive placement"],
    ]
    _write_custom_table(
        output_path=output_path,
        headers=["Environment", "Proxy signal", "Hidden objective", "Corruption channel", "Shift family"],
        rows=rows,
        caption=(
            "Environment instantiations used in the main paper. All tasks share the same restricted and rich "
            "trajectory-level monitor-channel protocol; the table highlights the task-specific proxy, hidden objective, corruption channel, and OOD shift family."
        ),
        label="tab:env-main",
        column_spec="p{1.9cm} p{2.2cm} p{2.6cm} p{2.6cm} p{3.3cm}",
        table_env="table*",
        size_command=r"\small",
    )


def main_results_compact_table(main_df: pd.DataFrame, output_path: str | Path) -> None:
    if main_df.empty:
        return
    rows: List[List[str]] = []
    midrules_after: List[int] = []
    order = ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"]
    for env_name in order:
        env_df = main_df[main_df["env"] == env_name].copy()
        if env_df.empty:
            continue
        env_df["method_order"] = env_df["method"].map({"PROXY_PPO": 0, "ORACLE_TRUE_PPO": 1})
        env_df = env_df.sort_values("method_order")
        start_index = len(rows)
        for row in env_df.itertuples(index=False):
            rows.append(
                [
                    ENV_LABELS.get(row.env, row.env) if len(rows) == start_index else "",
                    METHOD_LABELS.get(row.method, row.method),
                    f"{row.true_score_ID:.3f}",
                    f"{row.true_score_OOD:.3f}",
                    f"{row.hacking_incidence_ID:.3f}",
                    f"{row.hacking_incidence_OOD:.3f}",
                ]
            )
        midrules_after.append(len(rows) - 1)
    if midrules_after:
        midrules_after = midrules_after[:-1]
    _write_custom_table(
        output_path=output_path,
        headers=["Environment", "Policy", "ID true", "OOD true", "ID hack", "OOD hack"],
        rows=rows,
        caption="Main specification-gap results used in the paper narrative. Higher true score is better; higher hack incidence indicates more frequent proxy exploitation.",
        label="tab:main-results-compact",
        column_spec="p{1.8cm} p{2.05cm} c c c c",
        size_command=r"\small",
        midrules_after=midrules_after,
    )


def sensortamper_monitor_table(monitor_df: pd.DataFrame, output_path: str | Path) -> None:
    if monitor_df.empty:
        return
    subset = monitor_df[(monitor_df["env"] == "SensorTamper-v0") & (monitor_df["target"] == "hack")].copy()
    subset = subset[subset["leakage_status"].isin(["pass", "warning"])].dropna(subset=["ROC_AUC", "PR_AUC"])
    if subset.empty:
        return
    restricted = (
        subset[subset["channel"] == "restricted"]
        .set_index("model")[["ROC_AUC", "PR_AUC"]]
        .rename(columns={"ROC_AUC": "restricted_roc", "PR_AUC": "restricted_pr"})
    )
    rich = (
        subset[subset["channel"] == "rich"]
        .set_index("model")[["ROC_AUC", "PR_AUC"]]
        .rename(columns={"ROC_AUC": "rich_roc", "PR_AUC": "rich_pr"})
    )
    merged = restricted.join(rich, how="inner").reset_index()
    rows = []
    for row in merged.itertuples(index=False):
        rows.append(
            [
                MODEL_LABELS.get(row.model, row.model),
                f"{row.restricted_roc:.3f}",
                f"{row.restricted_pr:.3f}",
                f"{row.rich_roc:.3f}",
                f"{row.rich_pr:.3f}",
                f"{(row.rich_roc - row.restricted_roc):.3f}",
            ]
        )
    _write_custom_table(
        output_path=output_path,
        headers=["Model", "Restricted ROC", "Restricted PR", "Rich ROC", "Rich PR", "$\\Delta$ROC"],
        rows=rows,
        caption="Leakage-cleared hack-monitor results for SensorTamper-v0 under the proxy-only monitor protocol.",
        label="tab:sensortamper-monitor",
        column_spec="p{2.35cm} c c c c c",
        size_command=r"\small",
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
