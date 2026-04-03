"""Leakage diagnostics for monitor features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from src.utils.io_utils import ensure_dir

from .channels import channel_matrix


@dataclass
class LeakageResult:
    env_name: str
    channel: str
    target_name: str
    status: str
    max_single_feature_auc: float
    max_single_feature_name: str
    multifeature_auc: float | None
    source_prediction_auc: Dict[str, float]
    notes: List[str]


def _single_feature_auc(values: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(np.unique(values)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def run_leakage_checks(feature_rows: List[dict], env_name: str, channel_name: str, target_name: str) -> LeakageResult:
    labels = np.array([int(row[target_name]) for row in feature_rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return LeakageResult(env_name, channel_name, target_name, "degenerate", float("nan"), "", None, {}, ["single target class"])
    spec, matrix = channel_matrix(feature_rows, channel_name)
    max_auc = -1.0
    max_name = ""
    notes: List[str] = []
    for feature_name in spec.feature_names:
        values = np.array([float(row.get(feature_name, 0.0)) for row in feature_rows], dtype=np.float32)
        auc = _single_feature_auc(values, labels)
        if np.isnan(auc):
            continue
        auc = max(auc, 1.0 - auc)
        if auc > max_auc:
            max_auc = auc
            max_name = feature_name
    multifeature_auc = None
    x = np.array(matrix, dtype=np.float32)
    if len(feature_rows) >= 12:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=min(3, int(labels.sum()), int((labels == 0).sum())), shuffle=True, random_state=0)
        if cv.n_splits >= 2:
            probs = cross_val_predict(clf, x, labels, cv=cv, method="predict_proba")[:, 1]
            multifeature_auc = float(roc_auc_score(labels, probs))
    source_prediction_auc: Dict[str, float] = {}
    for source_name in ["source_tamper_used", "source_beacon_active", "source_gem_heavy"]:
        source = np.array([int(row.get(source_name, 0)) for row in feature_rows], dtype=np.int64)
        if len(np.unique(source)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=min(3, int(source.sum()), int((source == 0).sum())), shuffle=True, random_state=0)
        if cv.n_splits < 2:
            continue
        probs = cross_val_predict(clf, x, source, cv=cv, method="predict_proba")[:, 1]
        source_prediction_auc[source_name] = float(roc_auc_score(source, probs))
    status = "pass"
    if max_auc >= 0.98:
        status = "fail"
        notes.append(f"single feature {max_name} reached AUC {max_auc:.3f}")
    elif multifeature_auc is not None and multifeature_auc >= 0.995:
        status = "fail"
        notes.append(f"combined features reached AUC {multifeature_auc:.3f}")
    elif max_auc >= 0.93:
        status = "warning"
        notes.append(f"suspicious single-feature AUC {max_auc:.3f} from {max_name}")
    return LeakageResult(env_name, channel_name, target_name, status, max_auc, max_name, multifeature_auc, source_prediction_auc, notes)


def write_leakage_result(result: LeakageResult, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    ensure_dir(output)
    path = output / f"{result.env_name}_{result.channel}_{result.target_name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, sort_keys=True)
    return path

