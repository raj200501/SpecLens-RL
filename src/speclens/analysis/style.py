"""Shared matplotlib style for paper figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# Categorical slots, assigned in fixed order (validated for colour-vision deficiency).
METHOD_COLORS = {"oracle": "#2a78d6", "proxy": "#eb6834"}
METHOD_LABELS = {"oracle": "Oracle PPO (trained on $J_t$)", "proxy": "Proxy PPO (trained on $R_p$)"}
# Monitor channels are ordered by how much they may see: one-hue ordinal ramp.
CHANNEL_COLORS = {"behavior": "#86b6ef", "task": "#2a78d6", "privileged": "#104281"}
ENV_ORDER = ("ProxyFarming", "SensorTamper", "GemTimeout")


def envs_in(df) -> list[str]:
    """Tasks present in ``df``, in canonical order."""
    present = set(df["env"])
    return [env for env in ENV_ORDER if env in present]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, path_stem) -> None:
    """Write ``<stem>.pdf`` for the paper and ``<stem>.png`` for the README."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png")
    plt.close(fig)
