from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .config import LABELS


def compute_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def build_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
