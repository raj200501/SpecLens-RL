"""Monitor evaluation: grouped cross-validation and baselines.

The unit of independence is the training run (one PPO seed): episodes from the
same run share a policy lineage and are strongly correlated, so every split
holds out whole runs. A monitor's score on a held-out run is computed by a model
that never saw any episode of that run.
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


def label_status(y: np.ndarray) -> str:
    """``ok`` if both classes occur and the minority class is at least 5% of episodes."""
    if len(np.unique(y)) < 2 or min(y.mean(), 1 - y.mean()) < MIN_MINORITY_FRAC:
        return "degenerate"
    return "ok"


@dataclass
class CVResult:
    fold_auc: list[float]  # ROC-AUC on each held-out run that contains both classes
    oof_scores: np.ndarray  # out-of-fold monitor scores, aligned with the input rows (NaN if not scored)

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_auc)) if self.fold_auc else float("nan")


def grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str) -> CVResult:
    """Leave-one-run-out cross-validation.

    Every row is scored by a model trained on all other runs (when those contain
    both classes). Runs whose own episodes are all one class are scored but
    contribute no fold AUC.
    """
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for g in np.unique(groups):
        test = groups == g
        if len(np.unique(y[~test])) < 2:
            continue
        clf = build_model(model).fit(X[~test], y[~test])
        oof[test] = clf.predict_proba(X[test])[:, 1]
        if len(np.unique(y[test])) == 2:
            fold_auc.append(float(roc_auc_score(y[test], oof[test])))
    return CVResult(fold_auc=fold_auc, oof_scores=oof)


def stratified_auc(y: np.ndarray, scores: np.ndarray, strata: np.ndarray) -> float:
    """Mean ROC-AUC within strata that contain both classes.

    With strata = (run, snapshot), this asks whether the monitor separates
    hacking from non-hacking episodes *of the same policy*, which removes any
    credit for recognizing which policy, or how late in training, produced an
    episode.
    """
    aucs = [
        roc_auc_score(y[strata == s], scores[strata == s])
        for s in np.unique(strata)
        if len(np.unique(y[strata == s])) == 2
    ]
    return float(np.mean(aucs)) if aucs else float("nan")


def single_feature_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Held-out ROC-AUC of each feature used alone as a monitor score (descriptive).

    The feature's orientation is chosen on the training runs of each fold.
    Returns the mean over held-out runs for every column of ``X``.
    """
    aucs = np.full((len(np.unique(groups)), X.shape[1]), np.nan)
    for k, g in enumerate(np.unique(groups)):
        test = groups == g
        if len(np.unique(y[test])) < 2:
            continue
        for j in range(X.shape[1]):
            sign = 1.0 if _auc_or_half(y[~test], X[~test, j]) >= 0.5 else -1.0
            aucs[k, j] = _auc_or_half(y[test], sign * X[test, j])
    return np.nanmean(aucs, axis=0)


def selected_feature_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[list[float], list[int]]:
    """Best-single-feature baseline with the feature *selected on the training runs*.

    For each held-out run, the feature (and orientation) with the highest AUC on
    the other runs is chosen and then evaluated on the held-out run. Returns the
    per-fold AUCs and the chosen feature index of each fold.
    """
    fold_auc, chosen = [], []
    for g in np.unique(groups):
        test = groups == g
        if len(np.unique(y[test])) < 2 or len(np.unique(y[~test])) < 2:
            continue
        train_auc = np.array([_auc_or_half(y[~test], X[~test, j]) for j in range(X.shape[1])])
        j = int(np.argmax(np.abs(train_auc - 0.5)))
        sign = 1.0 if train_auc[j] >= 0.5 else -1.0
        fold_auc.append(_auc_or_half(y[test], sign * X[test, j]))
        chosen.append(j)
    return fold_auc, chosen


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


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    keep = ~np.isnan(scores)
    return float(average_precision_score(y[keep], scores[keep]))


def _auc_or_half(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or np.ptp(scores) == 0:
        return 0.5
    return float(roc_auc_score(y, scores))
