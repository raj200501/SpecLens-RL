"""Plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io_utils import ensure_dir

matplotlib.use("Agg")


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
