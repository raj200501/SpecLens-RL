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

MONITOR_STATUS_LABELS = {
    "accepted": "Accepted",
    "degenerate": "Degenerate",
    "rejected": "Rejected",
    "missing": "Missing",
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
    resize_to: str | None = None,
) -> None:
    target = Path(output_path)
    midrules_after = set(midrules_after or [])
    lines = [
        rf"\begin{{{table_env}}}[t]",
        r"\centering",
        size_command,
    ]
    if resize_to:
        lines.append(rf"\resizebox{{{resize_to}}}{{!}}{{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
    )
    for idx, row in enumerate(rows):
        lines.append(" & ".join(row) + r" \\")
        if idx in midrules_after:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize_to:
        lines.append(r"}")
    lines.extend(
        [
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


def environment_summary_compact_table(output_path: str | Path) -> None:
    shared_monitor = "Restricted trace; rich audit summary"
    rows = [
        [
            "ProxyFarming",
            "Beacon reward",
            "Reach goal efficiently",
            "Repeated beacon harvesting",
            shared_monitor,
            "Beacon move; layout; distractors",
        ],
        [
            "SensorTamper",
            "Sensor-mediated reward",
            "Deliver object to target",
            "Tamper tile spoofs sensor",
            shared_monitor,
            "Tamper move; delay; path change",
        ],
        [
            "GemTimeout",
            "Gem reward",
            "Reach goal before timeout",
            "Gem detours trade off with time",
            shared_monitor,
            "Density; budget; deceptive placement",
        ],
    ]
    _write_custom_table(
        output_path=output_path,
        headers=["Environment", "Proxy reward", "True objective", "Corruption channel", "Monitor channels", "OOD shift"],
        rows=rows,
        caption=(
            "Compact summary of the three benchmark environments used in the main paper."
        ),
        label="tab:env-summary-compact",
        column_spec="p{1.65cm} p{1.8cm} p{2.0cm} p{2.1cm} p{2.5cm} p{2.2cm}",
        table_env="table*",
        size_command=r"\small",
        resize_to=r"\textwidth",
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
        resize_to=r"\columnwidth",
    )


def sensortamper_monitor_table(monitor_df: pd.DataFrame, output_path: str | Path) -> None:
    if monitor_df.empty:
        return
    subset = monitor_df[(monitor_df["env"] == "SensorTamper-v0") & (monitor_df["target"] == "hack")].copy()
    subset = subset[subset["leakage_status"].isin(["pass", "warning"])].dropna(subset=["ROC_AUC", "PR_AUC"])
    if subset.empty:
        return
    rows = []
    channel_labels = {"restricted": "Restricted", "rich": "Rich"}
    for row in subset.sort_values(by=["channel", "model"]).itertuples(index=False):
        rows.append(
            [
                channel_labels.get(row.channel, row.channel),
                MODEL_LABELS.get(row.model, row.model),
                f"{row.ROC_AUC:.3f}",
                f"{row.PR_AUC:.3f}",
                row.leakage_status.capitalize(),
            ]
        )
    _write_custom_table(
        output_path=output_path,
        headers=["Channel", "Monitor", "ROC-AUC", "PR-AUC", "Leakage"],
        rows=rows,
        caption="Leakage-cleared hack-monitor results for SensorTamper-v0 under the proxy-only monitor protocol.",
        label="tab:sensortamper-monitor",
        column_spec="p{1.6cm} p{2.25cm} c c c",
        size_command=r"\small",
        resize_to=r"\columnwidth",
    )


def monitor_scope_table(scope_df: pd.DataFrame, output_path: str | Path) -> None:
    if scope_df.empty:
        return
    rows = []
    order = ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"]
    for env_name in order:
        subset = scope_df[scope_df["env"] == env_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        outcome = MONITOR_STATUS_LABELS.get(str(row["hack_monitor_status"]), str(row["hack_monitor_status"]).capitalize())
        if row["hack_monitor_channels"]:
            channel_text = str(row["hack_monitor_channels"]).replace("restricted", "restricted").replace("rich", "rich")
            outcome = f"{outcome} ({channel_text})"
        rows.append(
            [
                ENV_LABELS.get(env_name, env_name),
                f"{int(row['hack_positive'])}/{int(row['hack_total'])}",
                f"{int(row['failure_positive'])}/{int(row['failure_total'])}",
                outcome,
            ]
        )
    _write_custom_table(
        output_path=output_path,
        headers=["Environment", "Hack labels", "Failure labels", "Hack monitor task"],
        rows=rows,
        caption=(
            "Proxy-only ID monitor scope. Only SensorTamper-v0 yields a nondegenerate hack-label mix under the second-pass protocol; "
            "failure labels collapse in all three environments."
        ),
        label="tab:monitor-scope",
        column_spec="p{1.75cm} p{1.3cm} p{1.45cm} p{2.45cm}",
        size_command=r"\small",
        resize_to=r"\columnwidth",
    )


def seed_variability_table(variability_df: pd.DataFrame, output_path: str | Path) -> None:
    if variability_df.empty:
        return
    rows = []
    order = ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"]
    for env_name in order:
        env_df = variability_df[(variability_df["env"] == env_name) & (variability_df["split"] == "id")]
        if env_df.empty:
            continue
        proxy = env_df[env_df["method"] == "PROXY_PPO"]
        oracle = env_df[env_df["method"] == "ORACLE_TRUE_PPO"]
        if proxy.empty or oracle.empty:
            continue
        proxy_row = proxy.iloc[0]
        oracle_row = oracle.iloc[0]
        rows.append(
            [
                ENV_LABELS.get(env_name, env_name),
                f"{proxy_row['true_score_mean']:.3f} +/- {proxy_row['true_score_std']:.3f}",
                f"{proxy_row['hack_mean']:.3f} +/- {proxy_row['hack_std']:.3f}",
                f"{oracle_row['true_score_mean']:.3f} +/- {oracle_row['true_score_std']:.3f}",
            ]
        )
    _write_custom_table(
        output_path=output_path,
        headers=["Environment", "Proxy ID true", "Proxy ID hack", "Oracle ID true"],
        rows=rows,
        caption="ID seed variability across the three proxy and oracle runs. Entries are mean +/- standard deviation over seeds.",
        label="tab:seed-variability",
        column_spec="l c c c",
        size_command=r"\small",
    )


def seed_dominance_table(dominance_df: pd.DataFrame, output_path: str | Path) -> None:
    if dominance_df.empty:
        return
    rows = []
    split_labels = {"id": "ID", "ood": "OOD"}
    order = ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"]
    for env_name in order:
        env_df = dominance_df[dominance_df["env"] == env_name]
        if env_df.empty:
            continue
        env_df = env_df.copy()
        env_df["split_order"] = env_df["split"].map({"id": 0, "ood": 1}).fillna(99)
        for row in env_df.sort_values("split_order").itertuples(index=False):
            rows.append(
                [
                    ENV_LABELS.get(row.env, row.env),
                    split_labels.get(row.split, str(row.split).upper()),
                    f"{row.proxy_max_true_score:.3f}",
                    f"{row.oracle_min_true_score:.3f}",
                    f"{row.dominance_margin:.3f}",
                ]
            )
    _write_custom_table(
        output_path=output_path,
        headers=["Environment", "Split", "Best proxy seed", "Worst oracle seed", "Margin"],
        rows=rows,
        caption=(
            "Cross-seed dominance margins on true score. A positive margin means the best proxy seed "
            "still scores below the worst oracle seed for that environment and split."
        ),
        label="tab:seed-dominance",
        column_spec="l c c c c",
        size_command=r"\small",
        resize_to=r"\columnwidth",
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
