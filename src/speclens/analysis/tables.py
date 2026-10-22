"""LaTeX tables and the number macros cited in the paper text."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from speclens.analysis.style import ENV_ORDER, envs_in
from speclens.envs import ENVS

METHOD_NAMES = {"proxy": "Proxy PPO", "oracle": "Oracle PPO"}
CHANNEL_NAMES = {"behavior": "behavior", "task": "task", "privileged": "privileged$^\\dagger$"}


def fmt(value: float, digits: int = 2) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def fmt_ci(row: pd.Series, metric: str, digits: int = 2) -> str:
    lo, hi = fmt(row[metric + "_lo"], digits), fmt(row[metric + "_hi"], digits)
    return f"{fmt(row[metric], digits)} {{\\scriptsize [{lo}, {hi}]}}"


def _escape(text: str) -> str:
    return text.replace("%", "\\%").replace("_", "\\_").replace("->", "$\\rightarrow$").replace(">=", "$\\geq$")


def task_table(path: Path) -> None:
    lines = [
        "\\begin{tabular}{@{}p{0.13\\linewidth}p{0.27\\linewidth}p{0.27\\linewidth}p{0.25\\linewidth}@{}}",
        "\\toprule",
        "Task & Corruption channel $C$ & Shift $\\Delta$ (OOD) & Hack label \\\\",
        "\\midrule",
    ]
    for name in ENV_ORDER:
        spec = ENVS[name].spec_tuple
        lines.append(f"{name} & {_escape(spec.corruption_channel)} & {_escape(spec.shift)} & {_escape(spec.hack)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def spec_table(path: Path) -> None:
    """Full proxy reward, true objective and success definition of every task."""
    lines = [
        "\\begin{tabular}{@{}p{0.13\\linewidth}p{0.3\\linewidth}p{0.27\\linewidth}p{0.22\\linewidth}@{}}",
        "\\toprule",
        "Task & Proxy reward $R_p$ & True objective $J_t$ & Success \\\\",
        "\\midrule",
    ]
    for name in ENV_ORDER:
        spec = ENVS[name].spec_tuple
        cells = [_escape(x) for x in (spec.proxy_reward, spec.true_objective, spec.success)]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hyperparameter_table(cfg: dict, path: Path) -> None:
    rows = [(key.replace("_", " "), value) for key, value in cfg["ppo"].items()]
    rows += [
        ("seeds (main / sweep)", f"{cfg['main']['seeds']} / {cfg['sweep']['seeds']}"),
        ("policy snapshots per run", cfg["main"]["checkpoints"]),
        ("evaluation layouts per split", cfg["evaluation"]["episodes"]),
        ("channel scales (sweep)", ", ".join(f"{s:g}" for s in cfg["sweep"]["channel_scales"])),
        ("monitor episodes per snapshot and split", cfg["monitor"]["episodes_per_checkpoint"]),
    ]
    lines = ["\\begin{tabular}{@{}ll@{}}", "\\toprule", "Setting & Value \\\\", "\\midrule"]
    lines += [
        f"{name} & {value:,} \\\\" if isinstance(value, int) else f"{name} & {value} \\\\" for name, value in rows
    ]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sweep_table(summary: pd.DataFrame, path: Path) -> None:
    """Dose-response: ID hack and success rates per channel scale."""
    data = summary[summary.split == "id"]
    scales = sorted(data.channel_scale.unique())
    header = " & ".join(f"$\\lambda={s:g}$" for s in scales)
    lines = [
        f"\\begin{{tabular}}{{@{{}}ll{'c' * len(scales)}@{{}}}}",
        "\\toprule",
        f"Task & Metric & {header} \\\\",
        "\\midrule",
    ]
    for env in envs_in(data):
        for metric, label in (("hack", "hack rate"), ("success", "success")):
            cells = [fmt(data[(data.env == env) & (data.channel_scale == s)].iloc[0][metric]) for s in scales]
            lines.append(f"{env if metric == 'hack' else ''} & {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_results_table(summary: pd.DataFrame, path: Path) -> None:
    """Final-policy outcomes; ``summary`` comes from ``stats.summarize``."""
    lines = [
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "Task & Policy & Success $\\uparrow$ & Hack rate $\\downarrow$ & True return $J_t$ & Proxy return $R_p$ \\\\",
        "\\midrule",
    ]
    for split in ("id", "ood"):
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{split.upper()} layouts}}}} \\\\")
        for env in envs_in(summary):
            for method in ("proxy", "oracle"):
                row = summary[(summary.env == env) & (summary.method == method) & (summary.split == split)].iloc[0]
                cells = [fmt_ci(row, m) for m in ("success", "hack", "true_return", "proxy_return")]
                label = env if method == "proxy" else ""
                lines.append(f"{label} & {METHOD_NAMES[method]} & " + " & ".join(cells) + " \\\\")
        if split == "id":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def monitor_table(results: pd.DataFrame, path: Path, model: str = "logistic") -> None:
    data = results[(results.target == "hack") & (results.model == model)]
    lines = [
        "\\begin{tabular}{@{}llccccl@{}}",
        "\\toprule",
        "Task & Channel & Monitor AUC & Worst run & ID$\\rightarrow$OOD & Early$\\rightarrow$late "
        "& Best single feature (AUC) \\\\",
        "\\midrule",
    ]
    envs = envs_in(results)
    for env in envs:
        part = data[data.env == env]
        if part.empty:
            status = results[(results.env == env) & (results.target == "hack")].iloc[0]
            rate = f"{status.positive_rate:.0%}".replace("%", "\\%")
            lines.append(
                f"{env} ({rate} hacks) & \\multicolumn{{6}}{{l}}{{labels {status.label_status}: not evaluated}} \\\\"
            )
            continue
        for i, channel in enumerate(CHANNEL_NAMES):
            row = part[part.channel == channel].iloc[0]
            feature = row.best_feature.replace("_", "\\_")
            env_cell = f"{env} ({row.positive_rate:.0%} hacks)".replace("%", "\\%") if i == 0 else ""
            lines.append(
                f"{env_cell} & {CHANNEL_NAMES[channel]} & {fmt_ci(row, 'auc')} & {fmt(row.auc_min_fold)} & "
                f"{fmt(row.ood_auc)} & {fmt(row.late_auc)} & \\texttt{{{feature}}} ({fmt(row.best_feature_auc)}) \\\\"
            )
        if env != envs[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_numbers(entries: Iterable[tuple[str, str]], path: Path) -> None:
    """Write ``\\res{key}`` macros so the paper text never hand-copies a number."""
    lines = [
        "% Generated by `python -m speclens.report`. Do not edit by hand.",
        "\\makeatletter",
        "\\providecommand{\\res}[1]{\\@ifundefined{res@#1}{\\textbf{??#1??}}{\\csname res@#1\\endcsname}}",
        "\\makeatother",
    ]
    for key, value in entries:
        lines.append(f"\\expandafter\\def\\csname res@{key}\\endcsname{{{value}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def penalty_table(summary: pd.DataFrame, path: Path) -> None:
    """Outcomes of Proxy PPO trained against the behavior monitor."""
    lines = [
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Task & Hack rate & Flagged & Undetected hacks & Recall before & Recall after \\\\",
        "\\midrule",
    ]
    for env in envs_in(summary):
        row = summary[summary.env == env].iloc[0]
        cells = [fmt_ci(row, m) for m in ("hack", "flagged", "undetected_hack", "recall_before", "recall_after")]
        lines.append(f"{env} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
