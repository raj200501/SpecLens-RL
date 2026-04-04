"""Plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir

matplotlib.use("Agg")


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

METHOD_COLORS = {
    "PROXY_PPO": "#C85C3A",
    "ORACLE_TRUE_PPO": "#2F5C8F",
}

CHANNEL_COLORS = {
    "restricted": "#607D8B",
    "rich": "#2A9D8F",
}

SPLIT_HATCH = {
    "ID": "",
    "OOD": "//",
}


def _ordered_envs(frame: pd.DataFrame) -> list[str]:
    envs = [env for env in ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"] if env in set(frame["env_name"]) or env in set(frame["env"])]
    return envs


def framework_overview(output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axis("off")

    def add_box(x, y, w, h, title, body, fc, ec="#334155"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02 * w, y + h - 0.22 * h, title, fontsize=11, fontweight="bold", va="top", ha="left", color="#102A43")
        ax.text(x + 0.02 * w, y + 0.12 * h, body, fontsize=9.4, va="bottom", ha="left", color="#243B53")

    def add_arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#486581",
            )
        )

    ax.text(0.02, 0.96, r"$T = (M, R_p, J_t, C, M_{\mathrm{mon}}, \Delta, \Pi)$", fontsize=16, fontweight="bold", va="top", color="#102A43")
    ax.text(
        0.02,
        0.90,
        "SpecLens-RL separates task dynamics, training signal, hidden objective, corruption,\nmonitor access, shift family, and policy family before any result is computed.",
        fontsize=10,
        va="top",
        color="#334E68",
    )

    add_box(0.03, 0.20, 0.24, 0.48, "Task Schema", "M: symbolic task dynamics\nRp: proxy reward for learning\nJt: hidden true objective\nC: corruption or tampering channel", "#E8F1F8")
    add_box(0.31, 0.20, 0.25, 0.48, "Audit and Shift", "Mmon: restricted and rich monitor views\nΔ: deterministic ID/OOD variants\nΠ: proxy and oracle policy families", "#EEF7EE")
    add_box(0.60, 0.46, 0.16, 0.22, "Benchmark", "ProxyFarming\nSensorTamper\nGemTimeout", "#FFF4DD")
    add_box(0.60, 0.20, 0.16, 0.18, "Artifacts", "Saved runs\nLeakage checks\nGenerated tables and figures", "#FDECEC")
    add_box(0.80, 0.22, 0.17, 0.44, "Metric Suite", "Specification gap\nTampering risk\nMonitorability\nGeneralization under shift", "#F1EAFE")

    add_arrow(0.27, 0.44, 0.31, 0.44)
    add_arrow(0.56, 0.54, 0.60, 0.57)
    add_arrow(0.56, 0.31, 0.60, 0.29)
    add_arrow(0.76, 0.57, 0.80, 0.50)
    add_arrow(0.76, 0.29, 0.80, 0.38)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def proxy_vs_true_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    envs = list(df["env_name"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 3.5), squeeze=False)
    for ax, env_name in zip(axes[0], envs):
        subset = df[df["env_name"] == env_name]
        for method, group in subset.groupby("method"):
            ax.scatter(group["proxy_return"], group["true_score"], label=method, alpha=0.7)
        ax.set_title(env_name)
        ax.set_xlabel("Proxy return")
        ax.set_ylabel("True score")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main_results_panel(main_df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    envs = [env for env in ["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"] if env in set(main_df["env"])]
    fig, axes = plt.subplots(2, len(envs), figsize=(11.2, 5.8), sharey="row")
    bar_width = 0.28
    offsets = {"id": -0.16, "ood": 0.16}

    for col, env_name in enumerate(envs):
        env_df = main_df[main_df["env"] == env_name].copy()
        env_df["method_order"] = env_df["method"].map({"PROXY_PPO": 0, "ORACLE_TRUE_PPO": 1})
        env_df = env_df.sort_values("method_order")
        x = np.arange(len(env_df))

        for split_name, metric_name in [("id", "true_score_ID"), ("ood", "true_score_OOD")]:
            axes[0, col].bar(
                x + offsets[split_name],
                env_df[metric_name],
                width=bar_width,
                color=[METHOD_COLORS[m] for m in env_df["method"]],
                edgecolor="#243B53",
                linewidth=0.8,
                hatch=SPLIT_HATCH[split_name.upper()],
            )
        for split_name, metric_name in [("id", "hacking_incidence_ID"), ("ood", "hacking_incidence_OOD")]:
            axes[1, col].bar(
                x + offsets[split_name],
                env_df[metric_name],
                width=bar_width,
                color=[METHOD_COLORS[m] for m in env_df["method"]],
                edgecolor="#243B53",
                linewidth=0.8,
                hatch=SPLIT_HATCH[split_name.upper()],
            )

        axes[0, col].set_title(ENV_LABELS.get(env_name, env_name), fontsize=11, pad=8)
        axes[0, col].axhline(0.0, color="#9FB3C8", linewidth=0.9)
        axes[1, col].axhline(0.0, color="#9FB3C8", linewidth=0.9)
        axes[0, col].set_xticks(x)
        axes[0, col].set_xticklabels([])
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels([METHOD_LABELS[m] for m in env_df["method"]], fontsize=9)
        axes[0, col].grid(axis="y", alpha=0.18, linewidth=0.7)
        axes[1, col].grid(axis="y", alpha=0.18, linewidth=0.7)

    axes[0, 0].set_ylabel("True score", fontsize=10)
    axes[1, 0].set_ylabel("Hack incidence", fontsize=10)
    for ax in axes[0]:
        ax.set_ylim(-3.7, 1.15)
    for ax in axes[1]:
        ax.set_ylim(0.0, 1.05)

    legend_items = [
        Patch(facecolor="#B0BEC5", edgecolor="#243B53", hatch="", label="ID"),
        Patch(facecolor="#B0BEC5", edgecolor="#243B53", hatch="//", label="OOD"),
        Line2D([0], [0], color=METHOD_COLORS["PROXY_PPO"], lw=8, label="Proxy PPO"),
        Line2D([0], [0], color=METHOD_COLORS["ORACLE_TRUE_PPO"], lw=8, label="Oracle True PPO"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def sensortamper_monitorability(monitor_df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    subset = monitor_df[(monitor_df["env"] == "SensorTamper-v0") & monitor_df["leakage_status"].isin(["pass", "warning"])].copy()
    subset = subset.dropna(subset=["ROC_AUC", "PR_AUC"])
    if subset.empty:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ax.axis("off")
        ax.text(0.02, 0.92, "No accepted SensorTamper monitor results are available.", ha="left", va="top", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=False)
    metrics = [("ROC_AUC", "ROC-AUC"), ("PR_AUC", "PR-AUC")]
    model_order = ["logistic_regression", "gradient_boosting"]
    x = np.arange(len(model_order))
    width = 0.32

    for ax, (metric_key, title) in zip(axes, metrics):
        for idx, channel_name in enumerate(["restricted", "rich"]):
            channel_df = (
                subset[subset["channel"] == channel_name]
                .set_index("model")
                .reindex(model_order)
                .reset_index()
            )
            ax.bar(
                x + (-0.17 if idx == 0 else 0.17),
                channel_df[metric_key],
                width=width,
                color=CHANNEL_COLORS[channel_name],
                edgecolor="#243B53",
                linewidth=0.8,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in model_order], fontsize=9)
        ax.set_ylim(0.85, 1.0)
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)

    legend_items = [
        Patch(facecolor=CHANNEL_COLORS["restricted"], edgecolor="#243B53", label="Restricted monitor"),
        Patch(facecolor=CHANNEL_COLORS["rich"], edgecolor="#243B53", label="Rich monitor"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.5,
        -0.02,
        "Only leakage-cleared hack-monitor results are shown. ProxyFarming and GemTimeout were degenerate under the proxy-only monitor protocol.",
        ha="center",
        va="top",
        fontsize=8.6,
        color="#486581",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def hacking_incidence_bars(main_df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = range(len(main_df))
    ax.bar([i - 0.15 for i in x], main_df["hacking_incidence_ID"], width=0.3, label="ID")
    ax.bar([i + 0.15 for i in x], main_df["hacking_incidence_OOD"], width=0.3, label="OOD")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{row.env}\n{row.method}" for row in main_df.itertuples()], rotation=20, ha="right")
    ax.set_ylabel("Hacking incidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def tampering_summary(df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(df["method"], df["tamper_utilization_rate"])
    ax.set_ylabel("Tamper utilization")
    ax.set_title("SensorTamper-v0")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def monitorability_gap(df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    valid = df["monitorability_gap"].notna().any()
    if valid:
        labels = [f"{row['env']}:{row['model']}" for _, row in df.iterrows()]
        ax.bar(range(len(df)), df["monitorability_gap"].fillna(0.0))
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Rich - restricted ROC-AUC")
    else:
        statuses = df.groupby("env")["leakage_status"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()
        ax.axis("off")
        lines = ["No accepted monitorability-gap estimates were produced.", ""]
        for row in statuses.itertuples(index=False):
            lines.append(f"{row.env}: {row.leakage_status}")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generalization_gap(main_df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x = range(len(main_df))
    axes[0].bar([i - 0.15 for i in x], main_df["true_score_ID"], width=0.3, label="ID")
    axes[0].bar([i + 0.15 for i in x], main_df["true_score_OOD"], width=0.3, label="OOD")
    axes[0].set_title("True score")
    axes[1].bar([i - 0.15 for i in x], main_df["hacking_incidence_ID"], width=0.3, label="ID")
    axes[1].bar([i + 0.15 for i in x], main_df["hacking_incidence_OOD"], width=0.3, label="OOD")
    axes[1].set_title("Hacking incidence")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{row.env}\n{row.method}" for row in main_df.itertuples()], rotation=20, ha="right")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
