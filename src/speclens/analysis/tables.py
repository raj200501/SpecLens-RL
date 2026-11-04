"""LaTeX tables and the number macros cited in the paper text."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from speclens.analysis.style import ENV_ORDER, envs_in
from speclens.envs import ENVS
from speclens.rl import PPOConfig

METHOD_NAMES = {"proxy": "Proxy PPO", "oracle": "Oracle PPO"}
CHANNEL_NAMES = {"behavior": "behavior", "task": "task", "privileged": "privileged"}


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}".replace(f"-{0:.{digits}f}", f"{0:.{digits}f}")  # no "-0.00"


def fmt_ci(row: pd.Series, metric: str, digits: int = 2) -> str:
    lo, hi = fmt(row[metric + "_lo"], digits), fmt(row[metric + "_hi"], digits)
    return f"{fmt(row[metric], digits)} {{\\scriptsize [{lo}, {hi}]}}"


def _escape(text: str) -> str:
    return text.replace("%", "\\%").replace("_", "\\_").replace("->", "$\\rightarrow$").replace(">=", "$\\geq$")


def _ragged(*widths: float) -> str:
    """Ragged-right paragraph columns of the given fractions of the line width."""
    return "".join(f">{{\\raggedright\\arraybackslash}}p{{{w}\\linewidth}}" for w in widths)


def task_table(path: Path) -> None:
    lines = [
        "\\begin{tabular}{@{}" + _ragged(0.13, 0.28, 0.24, 0.23) + "@{}}",
        "\\toprule",
        "Task & Corruption channel $C$ & Shift $\\Delta$ (OOD) & Hack label \\\\",
        "\\midrule",
    ]
    for name in ENV_ORDER:
        spec = ENVS[name].task_spec
        lines.append(f"{name} & {_escape(spec.corruption_channel)} & {_escape(spec.shift)} & {_escape(spec.hack)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def spec_table(path: Path) -> None:
    """Full proxy reward, true objective and success definition of every task."""
    lines = [
        "\\begin{tabular}{@{}" + _ragged(0.13, 0.3, 0.26, 0.19) + "@{}}",
        "\\toprule",
        "Task & Proxy reward $R_p$ & True objective $J_t$ & Success \\\\",
        "\\midrule",
    ]
    for name in ENV_ORDER:
        spec = ENVS[name].task_spec
        cells = [_escape(x) for x in (spec.proxy_reward, spec.true_objective, spec.success)]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hyperparameter_table(cfg: dict, path: Path) -> None:
    """Every PPO setting actually used (including defaults the config does not set)."""
    rows = [(key.replace("_", " "), value) for key, value in PPOConfig(**cfg["ppo"]).as_dict().items()]
    pen = cfg["monitor_penalty"]
    rows += [
        ("seeds (main / sweep)", f"{cfg['main']['seeds']} / {cfg['sweep']['seeds']}"),
        ("policy snapshots per run", cfg["main"]["checkpoints"]),
        ("evaluation layouts per split", cfg["evaluation"]["episodes"]),
        ("layouts per snapshot (training curves)", cfg["evaluation"]["curve_episodes"]),
        ("channel scales (sweep)", ", ".join(f"{s:g}" for s in cfg["sweep"]["channel_scales"])),
        ("layouts for the exact proxy-optimal policy", cfg["sweep"]["optimal_layouts"]),
        ("channel scales (exact optimum)", ", ".join(f"{s:g}" for s in cfg["sweep"]["optimal_scales"])),
        ("monitor episodes per snapshot and split", cfg["monitor"]["episodes_per_checkpoint"]),
        ("monitored snapshots", f"every {cfg['monitor']['checkpoint_stride']}. of {cfg['main']['checkpoints']}"),
        ("monitor models", ", ".join(cfg["monitor"]["models"])),
        ("monitor penalty: channel, $\\beta$, seeds", f"{pen['channel']}, {pen['beta']:g}, {pen['seeds']}"),
        ("step limit per episode", ", ".join(f"{ENVS[n].default_max_steps} ({n})" for n in ENV_ORDER)),
    ]
    lines = ["\\begin{tabular}{@{}ll@{}}", "\\toprule", "Setting & Value \\\\", "\\midrule"]
    lines += [f"{name} & {_cell(value)} \\\\" for name, value in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def sweep_table(summary: pd.DataFrame, optimal: pd.DataFrame, path: Path) -> None:
    """Dose-response: ID hack and success rates per channel scale, for Proxy PPO and the exact proxy optimum."""
    data = summary[summary.split == "id"]
    scales = sorted(data.channel_scale.unique())
    header = " & ".join(f"$\\lambda={s:g}$" for s in scales)
    lines = [
        f"\\begin{{tabular}}{{@{{}}lll{'c' * len(scales)}@{{}}}}",
        "\\toprule",
        f"Task & Metric & Policy & {header} \\\\",
        "\\midrule",
    ]
    for env in envs_in(data):
        for metric, label in (("hack", "hack rate"), ("success", "success")):
            for name, source in (("Proxy PPO", data), ("proxy-optimal", optimal)):
                part = source[source.env == env]
                cells = [fmt(part[part.channel_scale == s].iloc[0][metric]) for s in scales]
                task = env if (metric, name) == ("hack", "Proxy PPO") else ""
                metric_label = label if name == "Proxy PPO" else ""
                lines.append(f"{task} & {metric_label} & {name} & " + " & ".join(cells) + " \\\\")
        if env != envs_in(data)[-1]:
            lines.append("\\midrule")
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


def monitor_table(results: pd.DataFrame, path: Path, model: str = "logistic", target: str = "hack") -> None:
    """Held-out monitor AUCs next to the baselines a monitor has to beat."""
    data = results[(results.target == target) & (results.model == model)]
    lines = [
        "\\begin{tabular}{@{}llccccccc@{}}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{Monitor AUC} & \\multicolumn{2}{c}{Baselines} & Gain over & "
        "\\multicolumn{2}{c}{Transfer} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}",
        "Task & Channel & pooled & within snapshot & policy identity & single feature & identity & ID$\\to$OOD "
        "& early$\\to$late \\\\",
        "\\midrule",
    ]
    envs = envs_in(results)
    for env in envs:
        part = data[data.env == env]
        status = results[(results.env == env) & (results.target == target)].iloc[0]
        rate = f"{status.positive_rate:.0%}".replace("%", "\\%")
        if part.empty:
            lines.append(
                f"{env} ({rate}) & \\multicolumn{{8}}{{l}}{{labels {status.label_status}: not evaluated}} \\\\"
            )
        for i, channel in enumerate(CHANNEL_NAMES if not part.empty else ()):
            row = part[part.channel == channel].iloc[0]
            cells = [
                fmt_ci(row, "auc", 3),
                fmt(row.within_snapshot_auc, 3),
                fmt(row.identity_auc, 3) if i == 0 else "",
                fmt(row.selected_feature_auc, 3),
                fmt_ci(row, "gain_over_identity", 3),
                fmt(row.ood_auc, 3),
                fmt(row.late_auc, 3),
            ]
            name = CHANNEL_NAMES[channel] + ("" if row.admissible else "$^\\dagger$")  # sees every label field
            lines.append(f"{f'{env} ({rate})' if i == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
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


def penalty_table(summary: pd.DataFrame, channel: str, path: Path) -> None:
    """Outcomes of Proxy PPO trained against a fitted hack monitor."""
    lines = [
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "& Hack & Harmful & Flagged & Undetected & \\multicolumn{2}{c}{Monitor recall on hacks} \\\\",
        "\\cmidrule(lr){6-7}",
        "Task & rate & hack rate & rate & hack rate & unpenalized & penalized \\\\",
        "\\midrule",
    ]
    for env in envs_in(summary):
        row = summary[summary.env == env].iloc[0]
        cells = [fmt_ci(row, m) for m in ("hack", "harmful_hack", "flagged", "undetected_hack", "recall_before")]
        runs = row.get("recall_after_runs")
        runs = 0 if pd.isna(runs) else int(runs)
        if runs == 0:
            cells.append("-- (no run)")
        elif runs == 1:
            cells.append(f"{fmt(row.recall_after)} (1 run)")
        else:
            cells.append(f"{fmt_ci(row, 'recall_after')} ({runs} runs)")
        lines.append(f"{env} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text(f"% {channel}-channel monitor\n" + "\n".join(lines) + "\n", encoding="utf-8")


FEATURE_NAMES = {
    "length_ratio": "episode length / step limit",
    "frac_up": "fraction of up actions",
    "frac_down": "fraction of down actions",
    "frac_left": "fraction of left actions",
    "frac_right": "fraction of right actions",
    "frac_interact": "fraction of interact actions",
    "action_entropy": "action entropy",
    "repeat_action_frac": "repeated actions",
    "reversal_frac": "move reversals",
    "stay_frac": "steps without moving",
    "longest_stay_ratio": "longest stay / length",
    "unique_cell_ratio": "distinct cells visited",
    "max_cell_share": "share of time on one cell",
    "net_displacement": "net displacement",
    "path_efficiency": "path efficiency",
}


def evasion_table(shifts: pd.DataFrame, path: Path, top: int = 3) -> None:
    """The features that moved the monitor's score most between unpenalized and penalized hacks."""
    lines = [
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{Mean over hacking episodes} & SD in monitor & Share of the \\\\",
        "\\cmidrule(lr){3-4}",
        "Task & Behavior feature & unpenalized & penalized & training data & change in logit \\\\",
        "\\midrule",
    ]
    envs = envs_in(shifts)
    for env in envs:
        part = shifts[shifts.env == env]
        first = part.iloc[0]
        label = f"{env} ({first.n_after} hacks, {first.runs_after} run{'s' if first.runs_after > 1 else ''})"
        for i, (_, row) in enumerate(
            part.reindex(part.contribution.abs().sort_values(ascending=False).index).head(top).iterrows()
        ):
            share = f"{100 * row.share:.0f}\\%"
            name = FEATURE_NAMES.get(row.feature, _escape(row.feature))
            cells = f"{fmt(row.before)} & {fmt(row.after)} & {fmt(row.scale, 3)} & {share}"
            lines.append(f"{label if i == 0 else ''} & {name} & {cells} \\\\")
        if env != envs[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
