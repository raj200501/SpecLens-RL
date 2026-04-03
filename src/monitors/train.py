"""Monitor training routines."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from .channels import channel_matrix
from .models import MonitorModelSpec


def train_monitor(feature_rows: List[dict], channel_name: str, model_name: str, target_name: str):
    labels = np.array([int(row[target_name]) for row in feature_rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return {"status": "degenerate", "reason": f"{target_name} has a single class"}
    spec, matrix = channel_matrix(feature_rows, channel_name)
    x = np.array(matrix, dtype=np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.3, random_state=0, stratify=labels
    )
    model = MonitorModelSpec(model_name).build()
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, 1]
    else:
        scores = model.decision_function(x_test)
    return {
        "status": "ok",
        "spec": spec,
        "model": model,
        "x_test": x_test,
        "y_test": y_test,
        "scores": scores,
    }

