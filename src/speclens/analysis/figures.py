"""Paper figures. Every figure is computed from the CSVs in ``results/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from speclens.analysis.stats import summarize
from speclens.analysis.style import (
    CHANNEL_COLORS,
    ENV_ORDER,
    INK,
    INK_SECONDARY,
    METHOD_COLORS,
    METHOD_LABELS,
    apply_style,
    envs_in,
    save,
)
from speclens.envs import GemTimeout, ProxyFarming, SensorTamper, make_env
from speclens.envs.base import bfs_distances

CELL_COLORS = {
    "#": "#3d3d3a",
    "G": "#1baf7a",
    "A": "#2a78d6",
    "B": "#eb6834",
    "C": "#eb6834",
    "*": "#eb6834",
    "O": "#9a7b4f",
}
CELL_GLYPHS = {"G": "G", "A": "A", "B": "B", "C": "C", "*": "◆", "O": "O"}
CHANNEL_NOTES = {
    "ProxyFarming": f"B = beacon (+{ProxyFarming.beacon_bonus:g} per step on it)",
    "SensorTamper": f"O = object, C = console (+{SensorTamper.false_confirmation:g} per false confirmation)",
    "GemTimeout": f"◆ = gem (+{GemTimeout.gem_bonus:g} each)",
}


def tasks_figure(path: Path, layout_seed: int = 7) -> None:
    """Example ID and OOD layouts for each task."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.4))
    for col, env_name in enumerate(ENV_ORDER):
        for row, split in enumerate(("id", "ood")):
            env = make_env(env_name, split=split)
            env.reset(seed=layout_seed)
            ax = axes[row, col]
            _draw_grid(ax, env.render().splitlines())
            extra = ""
            if env_name == "GemTimeout":
                extra = f", deadline {int(env.layout.params['deadline'])}"
            if env_name == "SensorTamper":
                console = env.layout.objects["console"][0]
                extra = f", console {bfs_distances(env.layout.agent, env.size, env.layout.walls)[console]} away"
            ax.set_title(f"{env_name} ({split.upper()}{extra})", fontsize=8, color=INK)
        axes[1, col].text(
            0.5,
            -0.08,
            CHANNEL_NOTES[env_name],
            transform=axes[1, col].transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color=INK_SECONDARY,
        )
    fig.tight_layout()
    save(fig, path)


def _draw_grid(ax, rows: list[str]) -> None:
    size = len(rows)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            face = CELL_COLORS.get(ch, "#f4f3ef")
            ax.add_patch(Rectangle((c, size - 1 - r), 0.94, 0.94, facecolor=face, edgecolor="none"))
            if ch in CELL_GLYPHS:
                ax.text(
                    c + 0.47,
                    size - 1 - r + 0.47,
                    CELL_GLYPHS[ch],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
    ax.set_xlim(-0.1, size)
    ax.set_ylim(-0.1, size)
    ax.set_aspect("equal")
    ax.axis("off")


def curves_figure(curves: pd.DataFrame, path: Path) -> None:
    """Proxy return, true return and hack rate over training for both methods.

    Thin lines are individual Proxy PPO runs, which in ProxyFarming switch to
    hacking at different times; the mean alone would hide that.
    """
    apply_style()
    data = curves[curves.channel_scale == 1.0]
    summary = summarize(data, ["env", "method", "env_steps"], ["proxy_return", "true_return", "hack"])
    panels = [("proxy_return", "Proxy return $R_p$"), ("true_return", "True return $J_t$"), ("hack", "Hack rate")]
    envs = envs_in(summary)
    fig, axes = plt.subplots(len(panels), len(envs), figsize=(7.0, 5.2), sharex=True, squeeze=False)
    for col, env in enumerate(envs):
        for row, (metric, label) in enumerate(panels):
            ax = axes[row, col]
            proxy_runs = data[(data.env == env) & (data.method == "proxy")]
            for _, run in proxy_runs.groupby("seed"):
                run = run.sort_values("env_steps")
                ax.plot(run.env_steps / 1e6, run[metric], color=METHOD_COLORS["proxy"], linewidth=0.5, alpha=0.35)
            for method in ("oracle", "proxy"):
                part = summary[(summary.env == env) & (summary.method == method)].sort_values("env_steps")
                x = part.env_steps / 1e6
                ax.plot(x, part[metric], color=METHOD_COLORS[method], label=METHOD_LABELS[method])
                ax.fill_between(
                    x, part[f"{metric}_lo"], part[f"{metric}_hi"], color=METHOD_COLORS[method], alpha=0.15, linewidth=0
                )
            if metric == "hack":
                ax.set_ylim(-0.03, 1.03)
            if row == 0:
                ax.set_title(env, color=INK)
            if col == 0:
                ax.set_ylabel(label)
            if row == len(panels) - 1:
                ax.set_xlabel("Environment steps (millions)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def dose_response_figure(episodes: pd.DataFrame, optimal: pd.DataFrame, sweep_seeds: int, path: Path) -> None:
    """Final hack and success rates as the corruption channel is scaled, next to the exact proxy optimum."""
    apply_style()
    proxy = episodes[(episodes.method == "proxy") & (episodes.split == "id") & (episodes.seed < sweep_seeds)]
    oracle = episodes[(episodes.method == "oracle") & (episodes.split == "id")]
    summary = summarize(proxy, ["env", "channel_scale"], ["hack", "success"])
    oracle_success = summarize(oracle, ["env"], ["success"]).set_index("env")["success"]
    envs = envs_in(summary)
    fig, axes = plt.subplots(2, len(envs), figsize=(7.0, 3.8), sharex=True, sharey=True, squeeze=False)
    color = METHOD_COLORS["proxy"]
    for col, env in enumerate(envs):
        part = summary[summary.env == env].sort_values("channel_scale")
        for row, (metric, label) in enumerate([("hack", "Hack rate"), ("success", "Success rate")]):
            ax = axes[row, col]
            ax.plot(
                part.channel_scale,
                part[metric],
                color=color,
                marker="o",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label="Proxy PPO",
            )
            ax.fill_between(
                part.channel_scale, part[f"{metric}_lo"], part[f"{metric}_hi"], color=color, alpha=0.15, linewidth=0
            )
            best = optimal[optimal.env == env].sort_values("channel_scale")
            ax.plot(
                best.channel_scale,
                best[metric],
                color=INK,
                linewidth=1.0,
                linestyle=(0, (1, 1.5)),
                marker="s",
                markersize=2.5,
                label="Proxy-optimal policy",
            )
            if metric == "success":
                ax.axhline(
                    oracle_success[env], color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)), label="Oracle PPO"
                )
            ax.set_ylim(-0.03, 1.03)
            if row == 0:
                ax.set_title(env, color=INK)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Corruption-channel scale")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def monitor_figure(results: pd.DataFrame, path: Path, model: str = "logistic") -> None:
    """Held-out hack-monitor AUC per channel against the baselines it has to beat.

    A dot plot on a zoomed axis: every monitor here scores far above chance, and
    what matters is the gap to the baselines. Each channel's held-out AUC (filled,
    with its 95% interval) is joined to its within-snapshot AUC (open).
    """
    apply_style()
    data = results[(results.target == "hack") & (results.model == model)]
    envs = envs_in(results)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    offsets = dict(zip(CHANNEL_COLORS, (-0.27, 0.0, 0.27)))
    for e, env in enumerate(envs):
        part = data[data.env == env]
        if part.empty:
            status = results[(results.env == env) & (results.target == "hack")].iloc[0]
            ax.text(e, 0.95, f"labels {status.label_status}\n({status.positive_rate:.0%} hacks)", ha="center",
                    va="center", fontsize=7, color=INK_SECONDARY)  # fmt: skip
            continue
        ax.hlines(
            part.identity_auc.iloc[0],
            e - 0.42,
            e + 0.42,
            color=INK_SECONDARY,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            label="policy-identity baseline",
            zorder=2,
        )
        for channel, dx in offsets.items():
            row = part[part.channel == channel].iloc[0]
            x, color = e + dx, CHANNEL_COLORS[channel]
            ax.vlines(x, row.within_snapshot_auc, row.auc, color=color, linewidth=1.0, alpha=0.6, zorder=3)
            ax.errorbar(
                x,
                row.auc,
                yerr=[[row.auc - row.auc_lo], [row.auc_hi - row.auc]],
                fmt="o",
                color=color,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                elinewidth=1.0,
                capsize=0,
                zorder=5,
                label=f"{channel} channel",
            )
            ax.plot(x, row.within_snapshot_auc, "o", markersize=5.5, markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.2, zorder=5, label="within one policy snapshot")  # fmt: skip
            ax.plot(x + 0.09, row.selected_feature_auc, "D", markersize=3.5, color=INK, zorder=6,
                    label="best single feature")  # fmt: skip
    ax.set_xticks(range(len(envs)), envs)
    ax.set_xlim(-0.5, len(envs) - 0.5)
    shown = data[["auc_lo", "selected_feature_auc", "within_snapshot_auc", "identity_auc"]]
    lowest = float(shown.min().min()) if not data.empty else 0.5
    ax.set_ylim(max(0.5, lowest - 0.02), 1.005)
    ax.set_ylabel("Hack-monitor ROC-AUC")
    ax.grid(axis="x", visible=False)
    handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
    order = [f"{c} channel" for c in CHANNEL_COLORS]
    order += ["within one policy snapshot", "best single feature", "policy-identity baseline"]
    order = [label for label in order if label in handles]
    legend = ax.legend(
        [handles[k] for k in order],
        order,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.3),
        handlelength=1.4,
        columnspacing=1.2,
    )
    if "within one policy snapshot" in order:  # the open marker's legend entry stands for every channel
        legend.legend_handles[order.index("within one policy snapshot")].set_markeredgecolor(INK_SECONDARY)
    fig.tight_layout()
    save(fig, path)


def penalty_figure(penalty_curves: pd.DataFrame, curves: pd.DataFrame, path: Path) -> None:
    """Hack rate and monitor-flag rate over training for agents penalized by the monitor (thin lines: runs)."""
    apply_style()
    pen = summarize(penalty_curves, ["env", "env_steps"], ["hack", "flagged"])
    base = summarize(curves[(curves.method == "proxy") & (curves.channel_scale == 1.0)], ["env", "env_steps"], ["hack"])
    envs = envs_in(pen)
    fig, axes = plt.subplots(1, len(envs), figsize=(7.0, 2.4), sharey=True, squeeze=False)
    series = [
        ("hack", "Hack rate (ground truth)", METHOD_COLORS["proxy"], "-"),
        ("flagged", "Flagged by the monitor", METHOD_COLORS["oracle"], "-"),
    ]
    for col, env in enumerate(envs):
        ax = axes[0, col]
        ref = base[base.env == env].sort_values("env_steps")
        ax.plot(
            ref.env_steps / 1e6,
            ref.hack,
            color=INK_SECONDARY,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            label="Hack rate without monitor penalty",
        )
        for _, run in penalty_curves[penalty_curves.env == env].groupby("seed"):
            run = run.sort_values("env_steps")
            ax.plot(run.env_steps / 1e6, run.hack, color=METHOD_COLORS["proxy"], linewidth=0.5, alpha=0.35)
        part = pen[pen.env == env].sort_values("env_steps")
        for metric, label, color, style in series:
            x = part.env_steps / 1e6
            ax.plot(x, part[metric], color=color, linestyle=style, label=label)
            ax.fill_between(x, part[f"{metric}_lo"], part[f"{metric}_hi"], color=color, alpha=0.15, linewidth=0)
        ax.set_title(env, color=INK)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Environment steps (millions)")
        if col == 0:
            ax.set_ylabel("Fraction of episodes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, path)
