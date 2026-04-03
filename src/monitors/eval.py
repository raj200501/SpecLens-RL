"""Monitor evaluation metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_monitor(train_result: Dict[str, object]) -> Dict[str, object]:
    if train_result["status"] != "ok":
        return train_result
    y_test = np.array(train_result["y_test"])
    scores = np.array(train_result["scores"])
    return {
        "status": "ok",
        "ROC_AUC": float(roc_auc_score(y_test, scores)),
        "PR_AUC": float(average_precision_score(y_test, scores)),
        "n_test": int(len(y_test)),
    }

