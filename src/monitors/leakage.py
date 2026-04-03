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


SUSPICIOUS_TOKENS = (
    "split",
    "reward",
    "proxy",
    "beacon",
    "tamper",
    "sensor",
    "gem",
    "risky",
    "timeout",
    "count",
    "presence",
    "activity",
    "success",
    "failure",
)


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
    suspicious_features_checked: List[str]
    max_behavioral_auc: float | None


def _single_feature_auc(values: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(np.unique(values)) < 2:
        return float("nan")
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def _safe_cv_auc(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2 or len(labels) < 12:
        return None
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    n_splits = min(3, n_pos, n_neg)
    if n_splits < 2:
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    probs = cross_val_predict(clf, matrix, labels, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, probs))


def _suspicious_features(spec) -> List[str]:
    names = set(spec.suspicious_features)
    for feature_name in spec.feature_names:
        if any(token in feature_name for token in SUSPICIOUS_TOKENS):
            names.add(feature_name)
    return sorted(names)


def run_leakage_checks(feature_rows: List[dict], env_name: str, channel_name: str, target_name: str) -> LeakageResult:
    labels = np.array([int(row[target_name]) for row in feature_rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return LeakageResult(
            env_name=env_name,
            channel=channel_name,
            target_name=target_name,
            status="degenerate",
            max_single_feature_auc=float("nan"),
            max_single_feature_name="",
            multifeature_auc=None,
            source_prediction_auc={},
            notes=["single target class"],
            suspicious_features_checked=[],
            max_behavioral_auc=None,
        )

    spec, matrix = channel_matrix(feature_rows, channel_name)
    suspicious_names = _suspicious_features(spec)
    max_auc = -1.0
    max_name = ""
    notes: List[str] = []

    for feature_name in suspicious_names:
        values = np.array([float(row.get(feature_name, 0.0)) for row in feature_rows], dtype=np.float32)
        auc = _single_feature_auc(values, labels)
        if np.isnan(auc):
            continue
        if auc > max_auc:
            max_auc = auc
            max_name = feature_name

    suspicious_matrix = None
    multifeature_auc = None
    if suspicious_names:
        suspicious_matrix = np.array(
            [[float(row.get(name, 0.0)) for name in suspicious_names] for row in feature_rows],
            dtype=np.float32,
        )
        multifeature_auc = _safe_cv_auc(suspicious_matrix, labels)

    full_matrix = np.array(matrix, dtype=np.float32)
    max_behavioral_auc = None
    if full_matrix.size > 0:
        max_behavioral_auc = _safe_cv_auc(full_matrix, labels)

    source_prediction_auc: Dict[str, float] = {}
    for source_name in ["source_tamper_used", "source_beacon_active", "source_gem_heavy"]:
        source = np.array([int(row.get(source_name, 0)) for row in feature_rows], dtype=np.int64)
        auc = _safe_cv_auc(full_matrix, source)
        if auc is not None:
            source_prediction_auc[source_name] = auc

    status = "pass"
    if suspicious_names and max_auc >= 0.98:
        status = "fail"
        notes.append(f"suspicious feature {max_name} reached AUC {max_auc:.3f}")
    elif multifeature_auc is not None and multifeature_auc >= 0.995:
        status = "fail"
        notes.append(f"suspicious feature bundle reached AUC {multifeature_auc:.3f}")
    else:
        if any(auc >= 0.995 for auc in source_prediction_auc.values()):
            status = "warning"
            notes.append("allowed behavior features strongly predict a label-source variable")
        if max_behavioral_auc is not None and max_behavioral_auc >= 0.995:
            notes.append("task appears behaviorally easy under the accepted channel")

    return LeakageResult(
        env_name=env_name,
        channel=channel_name,
        target_name=target_name,
        status=status,
        max_single_feature_auc=max_auc if max_auc >= 0 else float("nan"),
        max_single_feature_name=max_name,
        multifeature_auc=multifeature_auc,
        source_prediction_auc=source_prediction_auc,
        notes=notes,
        suspicious_features_checked=suspicious_names,
        max_behavioral_auc=max_behavioral_auc,
    )


def write_leakage_result(result: LeakageResult, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    ensure_dir(output)
    path = output / f"{result.env_name}_{result.channel}_{result.target_name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, sort_keys=True)
    return path
