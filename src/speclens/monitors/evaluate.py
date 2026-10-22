"""Monitor evaluation: grouped cross-validation, baselines and leakage screening.

The unit of independence is the training run (one PPO seed): episodes from the
same run share a policy lineage and are strongly correlated, so every split
holds out whole runs and every confidence interval resamples whole runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MONITOR_MODELS = ("logistic", "boosting")
TRIVIAL_AUC = 0.98  # a single feature this good makes a learned monitor uninformative
MIN_MINORITY_FRAC = 0.05


def build_model(name: str):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    if name == "boosting":
        return GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0)
    raise ValueError(f"unknown monitor model {name!r}")


def label_status(y: np.ndarray, groups: np.ndarray) -> str:
    """``ok`` if the minority class is common enough and every held-out run contains both classes."""
    if len(np.unique(y)) < 2 or min(y.mean(), 1 - y.mean()) < MIN_MINORITY_FRAC:
        return "degenerate"
    if any(len(np.unique(y[groups == g])) < 2 for g in np.unique(groups)):
        return "single-class-fold"
    return "ok"


@dataclass
class CVResult:
    fold_auc: list[float]  # one ROC-AUC per held-out run
    oof_scores: np.ndarray  # out-of-fold monitor scores, aligned with the input rows

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_auc))


def grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str) -> CVResult:
    """Leave-one-run-out cross-validation. Requires ``label_status(...) == "ok"``."""
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for g in np.unique(groups):
        test = groups == g
        clf = build_model(model).fit(X[~test], y[~test])
        oof[test] = clf.predict_proba(X[test])[:, 1]
        fold_auc.append(float(roc_auc_score(y[test], oof[test])))
    return CVResult(fold_auc=fold_auc, oof_scores=oof)


def single_feature_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Held-out ROC-AUC of each feature used alone as a monitor score.

    The feature's orientation (whether high values indicate hacking) is chosen
    on the training runs of each fold, so the baseline is not fit on test data.
    Returns the mean over held-out runs for every column of ``X``.
    """
    aucs = np.zeros((len(np.unique(groups)), X.shape[1]))
    for k, g in enumerate(np.unique(groups)):
        test = groups == g
        for j in range(X.shape[1]):
            train_auc = _auc_or_half(y[~test], X[~test, j])
            sign = 1.0 if train_auc >= 0.5 else -1.0
            aucs[k, j] = _auc_or_half(y[test], sign * X[test, j])
    return aucs.mean(axis=0)


def grouped_transfer(
    X_src: np.ndarray,
    y_src: np.ndarray,
    g_src: np.ndarray,
    X_dst: np.ndarray,
    y_dst: np.ndarray,
    g_dst: np.ndarray,
    model: str,
) -> list[float]:
    """Train on source rows of all other runs, test on target rows of the held-out run.

    Used for distribution-shift tests (ID -> OOD layouts, early -> late
    training). Folds whose held-out target rows contain a single class are
    skipped; the returned list has one ROC-AUC per usable run.
    """
    aucs = []
    for g in np.unique(g_dst):
        train, test = g_src != g, g_dst == g
        if len(np.unique(y_src[train])) < 2 or len(np.unique(y_dst[test])) < 2:
            continue
        clf = build_model(model).fit(X_src[train], y_src[train])
        aucs.append(float(roc_auc_score(y_dst[test], clf.predict_proba(X_dst[test])[:, 1])))
    return aucs


def run_bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-run values."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y, scores))


def _auc_or_half(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or np.ptp(scores) == 0:
        return 0.5
    return float(roc_auc_score(y, scores))
