"""Monitor training routines."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

from .channels import channel_matrix
from .models import MonitorModelSpec


def train_monitor(feature_rows: List[dict], channel_name: str, model_name: str, target_name: str):
    labels = np.array([int(row[target_name]) for row in feature_rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return {"status": "degenerate", "reason": f"{target_name} has a single class"}
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts[class_counts > 0] < 2):
        return {"status": "degenerate", "reason": f"{target_name} has too few minority examples for evaluation"}
    spec, matrix = channel_matrix(feature_rows, channel_name)
    x = np.array(matrix, dtype=np.float32)
    groups = np.array([row.get("run_id", f"row_{idx}") for idx, row in enumerate(feature_rows)], dtype=object)
    unique_groups = np.unique(groups)
    used_group_split = False
    cv_group_split = False
    if len(unique_groups) >= 3:
        logo = LeaveOneGroupOut()
        collected_indices = []
        collected_scores = []
        collected_labels = []
        for train_idx, test_idx in logo.split(x, labels, groups):
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            model = MonitorModelSpec(model_name).build()
            model.fit(x[train_idx], y_train)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(x[test_idx])[:, 1]
            else:
                scores = model.decision_function(x[test_idx])
            collected_indices.extend(test_idx.tolist())
            collected_scores.extend(np.asarray(scores, dtype=np.float32).tolist())
            collected_labels.extend(y_test.tolist())
        if collected_indices and len(np.unique(collected_labels)) >= 2:
            order = np.argsort(np.array(collected_indices))
            x_test = x[np.array(collected_indices)[order]]
            y_test = np.array(collected_labels, dtype=np.int64)[order]
            scores = np.array(collected_scores, dtype=np.float32)[order]
            used_group_split = True
            cv_group_split = True
    if not used_group_split and len(unique_groups) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.34, random_state=0)
        train_idx, test_idx = next(splitter.split(x, labels, groups=groups))
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            x_train = x[train_idx]
            x_test = x[test_idx]
            used_group_split = True
    if not used_group_split:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            labels,
            test_size=0.3,
            random_state=0,
            stratify=labels,
        )
    if not cv_group_split:
        model = MonitorModelSpec(model_name).build()
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)[:, 1]
        else:
            scores = model.decision_function(x_test)
    else:
        model = None
    return {
        "status": "ok",
        "spec": spec,
        "model": model,
        "x_test": x_test,
        "y_test": y_test,
        "scores": scores,
        "group_split": used_group_split,
        "group_cv": cv_group_split,
    }
