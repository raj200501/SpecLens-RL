"""Populate paper placeholders from generated artifacts when available."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io_utils import write_text


def write_experiment_section(results_root: str | Path, latex_root: str | Path) -> None:
    results_dir = Path(results_root)
    latex_root = Path(latex_root)
    latex_sections = Path(latex_root) / "sections"
    resource_csv = results_dir / "aggregated_resource_results.csv"
    if not resource_csv.exists():
        return
    resource_df = pd.read_csv(resource_csv)
    lines = [
        r"\section{Experimental Setup}",
        "We ran the saved experiment matrix using the active profile recorded in the preflight artifact.",
        "For monitor evaluation, we construct trajectory datasets from stochastic rollouts of the proxy-trained policy family only. The oracle baseline remains a diagnosis bound for specification-gap measurement rather than a deployable policy family, and mixing oracle with proxy rollouts made the first-pass monitor task method-trivial.",
        "Table \\ref{tab:resource-results} reports the concrete seed counts, step budgets, wall-clock usage, and disk footprint derived from saved run summaries.",
        r"\IfFileExists{tables/environment_summary.tex}{\input{tables/environment_summary}}{\todoartifact{environment summary table not yet generated}}",
        r"\IfFileExists{tables/resource_results.tex}{\input{tables/resource_results}}{\todoartifact{resource results table not yet generated}}",
        "",
    ]
    write_text(latex_sections / "experiments_placeholders.tex", "\n".join(lines))


def write_results_section(results_root: str | Path, latex_root: str | Path) -> None:
    results_dir = Path(results_root)
    latex_root = Path(latex_root)
    latex_sections = Path(latex_root) / "sections"
    main_csv = results_dir / "aggregated_main_results.csv"
    monitor_csv = results_dir / "aggregated_monitor_results.csv"
    if not main_csv.exists():
        return
    main_df = pd.read_csv(main_csv)
    if main_df.empty:
        return
    summary_sentence = make_empirical_summary(main_df, monitor_csv if monitor_csv.exists() else None)
    lines = [
        r"\section{Results}",
        summary_sentence,
        r"\IfFileExists{tables/main_specification_results.tex}{\input{tables/main_specification_results}}{\todoartifact{main specification table not yet generated}}",
        r"\IfFileExists{tables/tampering_results.tex}{\input{tables/tampering_results}}{\todoartifact{tampering table not yet generated}}",
        r"\IfFileExists{tables/monitor_results.tex}{\input{tables/monitor_results}}{\todoartifact{monitor table not yet generated}}",
        r"\begin{figure}[t]\centering\IfFileExists{figures/proxy_vs_true_scatter.pdf}{\includegraphics[width=\columnwidth]{figures/proxy_vs_true_scatter.pdf}}{\fbox{\parbox{0.95\columnwidth}{\centering Artifact placeholder: proxy\_vs\_true\_scatter.pdf}}}\caption{Proxy versus true-score relationship by environment and method.}\label{fig:proxy-true}\end{figure}",
        r"\begin{figure}[t]\centering\IfFileExists{figures/monitorability_gap.pdf}{\includegraphics[width=\columnwidth]{figures/monitorability_gap.pdf}}{\fbox{\parbox{0.95\columnwidth}{\centering Artifact placeholder: monitorability\_gap.pdf}}}\caption{Restricted versus rich monitor performance.}\label{fig:monitorability-gap}\end{figure}",
        "",
    ]
    write_text(latex_sections / "results_placeholders.tex", "\n".join(lines))
    write_text(latex_sections / "generated_abstract_results.tex", make_abstract_results_sentence(main_df, monitor_csv if monitor_csv.exists() else None))
    write_text(latex_sections / "generated_intro_results.tex", make_intro_results_sentence(main_df, monitor_csv if monitor_csv.exists() else None))
    write_text(latex_sections / "generated_conclusion_results.tex", make_conclusion_results_sentence(main_df, monitor_csv if monitor_csv.exists() else None))


def _compare_proxy_vs_oracle(main_df: pd.DataFrame):
    rows = []
    for env_name, group in main_df.groupby("env"):
        methods = set(group["method"])
        if "PROXY_PPO" not in methods or "ORACLE_TRUE_PPO" not in methods:
            continue
        proxy = group[group["method"] == "PROXY_PPO"].iloc[0]
        oracle = group[group["method"] == "ORACLE_TRUE_PPO"].iloc[0]
        rows.append(
            {
                "env": env_name,
                "proxy_true_id": float(proxy["true_score_ID"]),
                "oracle_true_id": float(oracle["true_score_ID"]),
                "proxy_hack_id": float(proxy["hacking_incidence_ID"]),
                "oracle_hack_id": float(oracle["hacking_incidence_ID"]),
                "proxy_true_ood": float(proxy["true_score_OOD"]),
                "oracle_true_ood": float(oracle["true_score_OOD"]),
            }
        )
    return rows


def _monitor_summary(monitor_csv: Path | None) -> str:
    if monitor_csv is None or not Path(monitor_csv).exists():
        return "Monitor results are not yet available."
    monitor_df = pd.read_csv(monitor_csv)
    if monitor_df.empty:
        return "Monitor results are not yet available."
    hack_df = monitor_df.copy()
    if "target" in hack_df.columns:
        hack_df = hack_df[hack_df["target"] == "hack"]
    score_column = "ROC_AUC"
    if score_column not in hack_df.columns and "ROC_AUC_hack" in hack_df.columns:
        score_column = "ROC_AUC_hack"
    accepted = sorted(
        hack_df.loc[
            hack_df[score_column].notna() & hack_df["leakage_status"].isin(["pass", "warning"]),
            "env",
        ].unique().tolist()
    )
    failures = sorted(hack_df.loc[hack_df["leakage_status"] == "fail", "env"].unique().tolist())
    degenerates = sorted(hack_df.loc[hack_df["leakage_status"] == "degenerate", "env"].unique().tolist())
    clauses = []
    if accepted:
        clauses.append(f"Leakage-controlled hack-monitor estimates were accepted for {', '.join(accepted)}")
    if failures:
        clauses.append(f"leakage diagnostics still failed for {', '.join(failures)}")
    if degenerates:
        clauses.append(f"hack-monitor targets remained degenerate for {', '.join(degenerates)}")
    if not clauses:
        return "Monitor evaluations are not yet available."
    return "; ".join(clauses) + "."


def make_empirical_summary(main_df: pd.DataFrame, monitor_csv: Path | None) -> str:
    comparisons = _compare_proxy_vs_oracle(main_df)
    failure_envs = [row["env"] for row in comparisons if row["proxy_true_id"] < row["oracle_true_id"] and row["proxy_hack_id"] > 0]
    mixed_envs = [row["env"] for row in comparisons if row["proxy_true_id"] >= row["oracle_true_id"] or row["proxy_hack_id"] == 0]
    parts = []
    if failure_envs:
        parts.append(f"Proxy-trained agents show clear specification failures in {', '.join(failure_envs)}.")
    if mixed_envs:
        parts.append(f"The empirical picture is mixed in {', '.join(sorted(set(mixed_envs)))}.")
    parts.append(_monitor_summary(monitor_csv))
    return " ".join(parts)


def make_abstract_results_sentence(main_df: pd.DataFrame, monitor_csv: Path | None) -> str:
    return " " + make_empirical_summary(main_df, monitor_csv) + "\n"


def make_intro_results_sentence(main_df: pd.DataFrame, monitor_csv: Path | None) -> str:
    return "\n\nAcross the saved second-pass runs, " + make_empirical_summary(main_df, monitor_csv) + "\n"


def make_conclusion_results_sentence(main_df: pd.DataFrame, monitor_csv: Path | None) -> str:
    return "\n\nEmpirically, " + make_empirical_summary(main_df, monitor_csv) + "\n"
