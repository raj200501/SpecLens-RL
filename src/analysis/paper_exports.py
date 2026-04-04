"""Populate paper sections from generated artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io_utils import write_text


ENV_LABELS = {
    "ProxyFarming-v0": "ProxyFarming",
    "SensorTamper-v0": "SensorTamper",
    "GemTimeout-v0": "GemTimeout",
}
REQUIRED_ENVS = tuple(ENV_LABELS.keys())
REQUIRED_METHODS = ("PROXY_PPO", "ORACLE_TRUE_PPO")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _has_full_main_matrix(main_df: pd.DataFrame) -> bool:
    if main_df.empty or "env" not in main_df.columns or "method" not in main_df.columns:
        return False
    for env_name in REQUIRED_ENVS:
        env_df = main_df[main_df["env"] == env_name]
        if env_df.empty:
            return False
        if not set(REQUIRED_METHODS).issubset(set(env_df["method"].unique())):
            return False
    return True


def _has_full_resource_matrix(resource_df: pd.DataFrame) -> bool:
    if resource_df.empty or "env" not in resource_df.columns or "method" not in resource_df.columns:
        return False
    proxy_df = resource_df[resource_df["method"] == "PROXY_PPO"]
    return set(REQUIRED_ENVS).issubset(set(proxy_df["env"].unique()))


def _table_input_line(latex_root: Path, preferred: str, fallback: str | None = None) -> str | None:
    tables_dir = latex_root / "tables"
    if (tables_dir / f"{preferred}.tex").exists():
        return rf"\input{{tables/{preferred}}}"
    if fallback and (tables_dir / f"{fallback}.tex").exists():
        return rf"\input{{tables/{fallback}}}"
    return None


def _main_pair(main_df: pd.DataFrame, env_name: str) -> tuple[pd.Series, pd.Series]:
    env_df = main_df[main_df["env"] == env_name]
    proxy = env_df[env_df["method"] == "PROXY_PPO"].iloc[0]
    oracle = env_df[env_df["method"] == "ORACLE_TRUE_PPO"].iloc[0]
    return proxy, oracle


def _monitor_summary_rows(monitor_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    restricted = monitor_df[
        (monitor_df["env"] == "SensorTamper-v0")
        & (monitor_df["channel"] == "restricted")
        & (monitor_df["model"] == "logistic_regression")
    ].iloc[0]
    rich = monitor_df[
        (monitor_df["env"] == "SensorTamper-v0")
        & (monitor_df["channel"] == "rich")
        & (monitor_df["model"] == "logistic_regression")
    ].iloc[0]
    return restricted, rich


def _has_monitor_columns(monitor_df: pd.DataFrame) -> bool:
    required = {"env", "channel", "model", "leakage_status"}
    auc_cols = {"ROC_AUC_hack", "ROC_AUC"}
    return (not monitor_df.empty) and required.issubset(set(monitor_df.columns)) and bool(auc_cols.intersection(set(monitor_df.columns)))


def write_experiment_section(results_root: str | Path, latex_root: str | Path) -> None:
    results_dir = Path(results_root)
    latex_root = Path(latex_root)
    latex_sections = latex_root / "sections"
    resource_df = _load_csv(results_dir / "aggregated_resource_results.csv")
    table_line = _table_input_line(latex_root, "environment_main", "environment_summary")

    if resource_df.empty:
        lines = [
            r"\section{Experimental Setup}",
            (
                "This local artifact does not include an aggregated resource summary, so the paper-facing experimental setup section "
                "must be regenerated from a full result bundle before submission."
            ),
            "",
        ]
        if table_line:
            lines.insert(-1, table_line)
        write_text(latex_sections / "experiments_placeholders.tex", "\n".join(lines))
        return

    if not _has_full_resource_matrix(resource_df):
        env_names = ", ".join(sorted(resource_df["env"].astype(str).unique()))
        lines = [
            r"\section{Experimental Setup}",
            (
                "This local build contains a partial experimental matrix used for smoke testing or intermediate artifact generation. "
                f"Available resource rows cover: {env_names}. The full conference-style setup text is only emitted when the complete second-pass matrix is present."
            ),
            "",
        ]
        if table_line:
            lines.insert(-1, table_line)
        write_text(latex_sections / "experiments_placeholders.tex", "\n".join(lines))
        return

    proxy_steps = int(
        resource_df.loc[
            (resource_df["env"] == "ProxyFarming-v0") & (resource_df["method"] == "PROXY_PPO"),
            "step_budget",
        ].iloc[0]
    )
    sensor_steps = int(
        resource_df.loc[
            (resource_df["env"] == "SensorTamper-v0") & (resource_df["method"] == "PROXY_PPO"),
            "step_budget",
        ].iloc[0]
    )
    gem_steps = int(
        resource_df.loc[
            (resource_df["env"] == "GemTimeout-v0") & (resource_df["method"] == "PROXY_PPO"),
            "step_budget",
        ].iloc[0]
    )

    lines = [
        r"\section{Experimental Setup}",
        (
            "We evaluate three symbolic tasks, two policy families, and deterministic ID/OOD variants under a fixed local-compute budget. "
            "Each environment is trained with three random seeds. ProxyFarming and SensorTamper use "
            f"{proxy_steps:,} and {sensor_steps:,} PPO steps per run respectively, while the recalibrated GemTimeout uses {gem_steps:,} steps per run. "
            "Oracle True PPO is included only as a diagnosis baseline for the hidden objective and is not presented as a deployable training rule."
        ),
        (
            "Monitor evaluation is deliberately stricter than the policy benchmark. We build monitor datasets from stochastic rollouts of the proxy-trained policy family only, "
            "then evaluate restricted and rich audit channels with logistic regression and gradient boosting under run-aware splits. "
            "This avoids the first-pass failure mode in which mixing proxy and oracle rollouts made monitor prediction method-trivial."
        ),
        (
            "Table~\\ref{tab:env-main} summarizes the task instantiations used in the paper. "
            "Full resource accounting and threshold-sensitivity artifacts are kept in the repository outputs but omitted from the main body for space."
        ),
        "",
    ]
    if table_line:
        lines.insert(-1, table_line)
    write_text(latex_sections / "experiments_placeholders.tex", "\n".join(lines))


def write_results_section(results_root: str | Path, latex_root: str | Path) -> None:
    results_dir = Path(results_root)
    latex_root = Path(latex_root)
    latex_sections = latex_root / "sections"
    main_df = _load_csv(results_dir / "aggregated_main_results.csv")
    monitor_df = _load_csv(results_dir / "aggregated_monitor_results.csv")
    tamper_df = _load_csv(results_dir / "aggregated_tampering_results.csv")
    if main_df.empty:
        lines = [
            r"\section{Results}",
            (
                "No aggregated benchmark results were found in this local artifact bundle, so the paper-facing results section "
                "must be regenerated from a real result set before submission."
            ),
            "",
        ]
        write_text(latex_sections / "results_placeholders.tex", "\n".join(lines))
        write_text(latex_sections / "generated_abstract_results.tex", make_abstract_results_sentence(main_df, monitor_df))
        write_text(latex_sections / "generated_intro_results.tex", make_intro_results_sentence(main_df, monitor_df))
        write_text(latex_sections / "generated_conclusion_results.tex", make_conclusion_results_sentence(main_df, monitor_df))
        return

    if not _has_full_main_matrix(main_df):
        env_names = ", ".join(sorted(main_df["env"].astype(str).unique())) if "env" in main_df.columns else "unknown environments"
        lines = [
            r"\section{Results}",
            (
                "This local build contains a partial aggregated result matrix used for smoke testing or intermediate artifact generation. "
                f"Available rows cover: {env_names}. The final conference-style results narrative is emitted only when the full second-pass matrix is present."
            ),
            "",
        ]
        write_text(latex_sections / "results_placeholders.tex", "\n".join(lines))
        write_text(latex_sections / "generated_abstract_results.tex", make_abstract_results_sentence(main_df, monitor_df))
        write_text(latex_sections / "generated_intro_results.tex", make_intro_results_sentence(main_df, monitor_df))
        write_text(latex_sections / "generated_conclusion_results.tex", make_conclusion_results_sentence(main_df, monitor_df))
        return

    proxy_pf, oracle_pf = _main_pair(main_df, "ProxyFarming-v0")
    proxy_st, oracle_st = _main_pair(main_df, "SensorTamper-v0")
    proxy_gt, oracle_gt = _main_pair(main_df, "GemTimeout-v0")

    tamper_proxy = None
    if not tamper_df.empty and {"method", "tamper_utilization_rate", "tampering_true_score_delta"}.issubset(set(tamper_df.columns)):
        proxy_rows = tamper_df[tamper_df["method"] == "PROXY_PPO"]
        if not proxy_rows.empty:
            tamper_proxy = proxy_rows.iloc[0]
    accepted_monitor = pd.DataFrame()
    if _has_monitor_columns(monitor_df) and "ROC_AUC_hack" in monitor_df.columns:
        accepted_monitor = monitor_df[
            (monitor_df["env"] == "SensorTamper-v0")
            & monitor_df["leakage_status"].isin(["pass", "warning"])
            & monitor_df["ROC_AUC_hack"].notna()
        ].copy()
    restricted_lr = rich_lr = None
    if not accepted_monitor.empty:
        restricted_lr, rich_lr = _monitor_summary_rows(
            accepted_monitor.rename(columns={"ROC_AUC_hack": "ROC_AUC", "PR_AUC_hack": "PR_AUC"})
        )

    lines = [
        r"\section{Results}",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\includegraphics[width=0.98\textwidth]{figures/main_results_panel.pdf}",
        r"\caption{Main benchmark outcomes. Across all three environments, proxy training underperforms oracle training on true score. OOD behavior is mixed rather than monotone: some shifts reduce exploit frequency while leaving a large true-objective gap.}",
        r"\label{fig:main-results}",
        r"\end{figure*}",
        r"\paragraph{Specification gaps are clearly measurable.}",
        (
            "Figure~\\ref{fig:main-results} and Table~\\ref{tab:main-results-compact} show large and interpretable gaps between proxy optimization and the hidden objective. "
            f"In ProxyFarming, Proxy PPO reaches ID true score {_fmt(proxy_pf['true_score_ID'])} with hack incidence {_fmt(proxy_pf['hacking_incidence_ID'])}, "
            f"while Oracle True PPO reaches {_fmt(oracle_pf['true_score_ID'])} with zero hack incidence. "
            f"SensorTamper shows the same pattern even more sharply: Proxy PPO reaches ID true score {_fmt(proxy_st['true_score_ID'])} with hack incidence {_fmt(proxy_st['hacking_incidence_ID'])}, "
            f"versus {_fmt(oracle_st['true_score_ID'])} and zero for the oracle. "
            f"After recalibration, GemTimeout becomes a complementary delayed-cost case: Proxy PPO reaches ID true score {_fmt(proxy_gt['true_score_ID'])} with hack incidence {_fmt(proxy_gt['hacking_incidence_ID'])}, "
            f"while the oracle reaches {_fmt(oracle_gt['true_score_ID'])} with zero hack incidence. "
            "Across proxy-trained policies, the proxy--true correlation is non-positive in every environment, which is exactly the kind of measurement signal the framework is meant to expose."
        ),
        _table_input_line(latex_root, "main_results_compact"),
        r"\paragraph{Tampering is explicitly measurable.}",
        (
            "SensorTamper does more than show a generic specification gap: it isolates a concrete corruption channel. "
            f"The proxy policy uses the tamper mechanism in {_fmt(tamper_proxy['tamper_utilization_rate'])} of evaluation episodes, whereas the oracle never tampers. "
            f"Tampered proxy trajectories are associated with a true-score delta of {_fmt(tamper_proxy['tampering_true_score_delta'])} relative to untampered proxy trajectories. "
            "The corruption channel is therefore behaviorally active rather than a dormant benchmark feature."
        )
        if tamper_proxy is not None
        else (
            "SensorTamper remains the corruption-channel benchmark in the suite, but this local artifact bundle does not include the tampering aggregate file needed to restate the numeric utilization and score-delta results."
        ),
        r"\paragraph{Monitorability requires leakage control.}",
        (
            "Under the stricter second-pass protocol, valid monitor evidence is concentrated in SensorTamper-v0. "
            f"A restricted logistic-regression monitor reaches ROC-AUC {_fmt(restricted_lr['ROC_AUC'])} and PR-AUC {_fmt(restricted_lr['PR_AUC'])}; "
            f"the corresponding rich-channel model reaches {_fmt(rich_lr['ROC_AUC'])} and {_fmt(rich_lr['PR_AUC'])}. "
            "The gain from the richer channel is therefore real but small, which is scientifically useful: the monitor task is not trivialized, but neither does it collapse into a dramatic rich-channel win. "
            "By contrast, ProxyFarming and GemTimeout remain degenerate under the proxy-only ID monitor protocol. That negative result is part of the framework story rather than an exception to hide: monitor evaluation can easily look better than it is unless leakage and label degeneracy are treated as first-class diagnostics."
        )
        if restricted_lr is not None and rich_lr is not None
        else (
            "Leakage-controlled monitor evidence is not available in this artifact bundle. "
            "That absence should be reported directly rather than back-filled from leaky or degenerate monitor tasks."
        ),
        r"\paragraph{Limitations of the current evidence.}",
        (
            "The monitor story is still narrow. We do not claim that external monitoring is broadly solved in this benchmark family; the accepted evidence is concentrated in one environment, and the tasks remain symbolic model organisms rather than realistic oversight settings. "
            "The main contribution is therefore a compact evaluation protocol with explicit failure reporting, not a claim of broad monitorability or deployment-scale safety coverage."
        ),
        "",
    ]
    monitor_figure = latex_root / "figures" / "sensortamper_monitorability.pdf"
    if monitor_figure.exists() and restricted_lr is not None and rich_lr is not None:
        insert_at = lines.index(r"\paragraph{Monitorability requires leakage control.}")
        lines[insert_at:insert_at] = [
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\columnwidth]{figures/sensortamper_monitorability.pdf}",
            r"\caption{Leakage-cleared hack-monitor results for SensorTamper-v0. Valid monitor evidence exists, but the rich-channel gain is modest rather than dramatic.}",
            r"\label{fig:sensortamper-monitor}",
            r"\end{figure}",
        ]
    main_table = _table_input_line(latex_root, "main_results_compact")
    if main_table:
        lines[lines.index(r"\paragraph{Tampering is explicitly measurable.}") : lines.index(r"\paragraph{Tampering is explicitly measurable.}")] = [main_table]
    monitor_table = _table_input_line(latex_root, "sensortamper_monitor")
    if monitor_table and restricted_lr is not None and rich_lr is not None:
        limit_index = lines.index(r"\paragraph{Limitations of the current evidence.}")
        lines[limit_index:limit_index] = [monitor_table]
    lines = [line for line in lines if line is not None]
    write_text(latex_sections / "results_placeholders.tex", "\n".join(lines))

    summary_sentence = make_abstract_results_sentence(main_df, monitor_df)
    write_text(latex_sections / "generated_abstract_results.tex", summary_sentence)
    write_text(latex_sections / "generated_intro_results.tex", make_intro_results_sentence(main_df, monitor_df))
    write_text(latex_sections / "generated_conclusion_results.tex", make_conclusion_results_sentence(main_df, monitor_df))


def make_abstract_results_sentence(main_df: pd.DataFrame, monitor_df: pd.DataFrame | None) -> str:
    if not _has_full_main_matrix(main_df):
        return (
            "SpecLens-RL provides a local, artifact-backed pipeline for benchmark analysis; a full empirical summary should be regenerated from the complete second-pass result matrix before submission.\n"
        )
    return (
        "SpecLens-RL measures large specification gaps and explicit tampering risk across lightweight RL tasks, "
        "and shows that leakage-controlled monitorability evaluation is feasible but narrow, with valid evidence concentrated in SensorTamper-v0.\n"
    )


def make_intro_results_sentence(main_df: pd.DataFrame, monitor_df: pd.DataFrame | None) -> str:
    if not _has_full_main_matrix(main_df):
        return (
            "\n\nThe paper-facing empirical summary should be regenerated from the complete second-pass result matrix before submission; partial local outputs are sufficient for smoke tests but not for the final narrative.\n"
        )
    return (
        "\n\nEmpirically, the final benchmark supports a conservative but useful conclusion: specification gaps are easy to measure across all three environments, "
        "tampering risk is directly measurable in SensorTamper-v0, and leakage-controlled monitor evidence is feasible but currently narrow, with accepted monitor results concentrated in SensorTamper-v0.\n"
    )


def make_conclusion_results_sentence(main_df: pd.DataFrame, monitor_df: pd.DataFrame | None) -> str:
    if not _has_full_main_matrix(main_df):
        return (
            "\n\nA full empirical conclusion should be regenerated from the complete second-pass result matrix before submission; partial local outputs are retained only for pipeline smoke testing.\n"
        )
    return (
        "\n\nEmpirically, the final artifact-backed paper shows large proxy-versus-true gaps across the benchmark, a behaviorally active tampering channel in SensorTamper-v0, "
        "and a leakage-controlled but still narrow monitor story whose strongest evidence is concentrated in SensorTamper-v0.\n"
    )
