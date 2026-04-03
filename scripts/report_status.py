#!/usr/bin/env python3
"""Write the final status report from existing artifacts."""

from __future__ import annotations

from datetime import datetime, timezone

from _bootstrap import ROOT
from src.analysis.aggregate import aggregate_episode_metrics, aggregate_monitor_results, aggregate_run_summaries
from src.utils.budget import directory_size_bytes
from src.utils.io_utils import read_json, write_text


def main() -> None:
    results_root = ROOT / "results"
    preflight = read_json(results_root / "preflight.json") if (results_root / "preflight.json").exists() else {}
    episodes = aggregate_episode_metrics(results_root)
    runs = aggregate_run_summaries(results_root)
    monitors = aggregate_monitor_results(results_root)
    stage1 = read_json(results_root / "stage1_summary.json") if (results_root / "stage1_summary.json").exists() else {}
    stage0 = read_json(results_root / "stage0_summary.json") if (results_root / "stage0_summary.json").exists() else {}
    stage2 = read_json(results_root / "stage2_summary.json") if (results_root / "stage2_summary.json").exists() else {}
    leakage_dir = results_root / "leakage"
    leakage_files = sorted(leakage_dir.glob("*.json"))
    if not monitors.empty and "leakage_status" in monitors.columns:
        counts = monitors["leakage_status"].fillna("missing").value_counts().to_dict()
        leakage_summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    else:
        leakage_summary = f"{len(leakage_files)} leakage diagnostic files present"
    lines = [
        "# Final Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Active profile: `{preflight.get('selected_profile', 'unknown')}`",
        f"- What was implemented: schema, environments, PPO core, monitor channels, leakage diagnostics, analysis, LaTeX scaffold, scripts, and tests.",
        f"- What actually ran: stage0=`{stage0.get('stage', 'not-run')}`, stage1 status=`{stage1.get('status', 'not-run')}`, stage2 status=`{stage2.get('status', 'not-run')}`.",
        f"- What was skipped: stage2 optional extensions unless explicitly launched.",
        f"- Why it was skipped: {stage2.get('reason', 'stage2 was not requested')}",
        f"- Runtime total: `{runs['wall_clock_sec'].sum() if not runs.empty and 'wall_clock_sec' in runs.columns else 0.0}` seconds across saved training summaries.",
        f"- Disk used in results/: `{directory_size_bytes(results_root)}` bytes.",
        f"- Override log summary: see `results/OVERRIDES.md`.",
        f"- Leakage diagnostic summary: {leakage_summary}.",
        f"- Exact reproducibility commands: `make preflight`, `make smoke`, `make stage1`, `make plots`, `make tables`, `make paper`, `make report`.",
        f"- What remains for a human to polish before submission: bibliography verification, paper wording after real results, environment calibration review, and any Stage 2 decision.",
        f"- Saved run count: `{len(runs)}`",
        f"- Saved episode count: `{len(episodes)}`",
        f"- Saved monitor result rows: `{len(monitors)}`",
    ]
    write_text(results_root / "FINAL_REPORT.md", "\n".join(lines) + "\n")
    print("results/FINAL_REPORT.md")


if __name__ == "__main__":
    main()
