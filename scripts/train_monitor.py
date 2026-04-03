#!/usr/bin/env python3
"""Build monitor datasets, run leakage diagnostics, and train monitors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from src.monitors.dataset import build_monitor_dataset
from src.monitors.eval import evaluate_monitor
from src.monitors.leakage import run_leakage_checks, write_leakage_result
from src.monitors.train import train_monitor
from src.utils.io_utils import append_inventory, write_csv_gz


def run_monitor_pipeline(
    env_name: str,
    results_root: Path | None = None,
    eval_split: str = "id",
    methods: list[str] | None = None,
) -> list[dict]:
    results_root = Path(results_root) if results_root is not None else (ROOT / "results")
    all_rows = build_monitor_dataset(results_root, env_name)
    dataset_rows = [row for row in all_rows if row["split"] == eval_split]
    if methods:
        allowed = set(methods)
        dataset_rows = [row for row in dataset_rows if row["method"] in allowed]
    if dataset_rows:
        write_csv_gz(results_root / "monitor_datasets" / f"{env_name}.csv.gz", dataset_rows)
        append_inventory(results_root / "artifact_inventory.csv", results_root / "monitor_datasets" / f"{env_name}.csv.gz", "monitor_dataset")
    monitor_rows = []
    for channel in ["restricted", "rich"]:
        leakage_hack = run_leakage_checks(dataset_rows, env_name, channel, "label_hack")
        write_leakage_result(leakage_hack, results_root / "leakage")
        leakage_failure = run_leakage_checks(dataset_rows, env_name, channel, "label_failure")
        write_leakage_result(leakage_failure, results_root / "leakage")
        for model_name in ["logistic_regression", "gradient_boosting"]:
            for target_name, target_label in [("label_hack", "hack"), ("label_failure", "failure")]:
                leakage = leakage_hack if target_name == "label_hack" else leakage_failure
                if leakage.status == "fail":
                    monitor_rows.append(
                        {
                            "env": env_name,
                            "channel": channel,
                            "model": model_name,
                            "target": target_label,
                            "eval_split": eval_split,
                            "ROC_AUC": None,
                            "PR_AUC": None,
                            "leakage_status": "fail",
                        }
                    )
                    continue
                train_result = train_monitor(dataset_rows, channel, model_name, target_name)
                metrics = evaluate_monitor(train_result)
                monitor_status = leakage.status if train_result.get("status") == "ok" else train_result.get("status", leakage.status)
                monitor_rows.append(
                    {
                        "env": env_name,
                        "channel": channel,
                        "model": model_name,
                        "target": target_label,
                        "eval_split": eval_split,
                        "ROC_AUC": metrics.get("ROC_AUC"),
                        "PR_AUC": metrics.get("PR_AUC"),
                        "leakage_status": monitor_status,
                    }
                )
    monitor_dir = results_root / "monitors"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    output = monitor_dir / f"{env_name}.json"
    output.write_text(json.dumps(monitor_rows, indent=2), encoding="utf-8")
    append_inventory(results_root / "artifact_inventory.csv", output, "monitor_results")
    return monitor_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--results-root")
    parser.add_argument("--eval-split", default="id")
    parser.add_argument("--methods")
    args = parser.parse_args()
    rows = run_monitor_pipeline(
        args.env,
        results_root=Path(args.results_root) if args.results_root else None,
        eval_split=args.eval_split,
        methods=[token.strip() for token in args.methods.split(",") if token.strip()] if args.methods else None,
    )
    print(pd.DataFrame(rows).to_json(orient="records"))


if __name__ == "__main__":
    main()
