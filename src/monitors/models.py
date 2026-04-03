"""Small monitor models."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class MonitorModelSpec:
    name: str

    def build(self):
        if self.name == "logistic_regression":
            return LogisticRegression(max_iter=2000, class_weight="balanced")
        if self.name == "gradient_boosting":
            return GradientBoostingClassifier(random_state=0)
        raise ValueError(f"Unsupported monitor model: {self.name}")

