from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score


@dataclass
class HybridWeights:
    classical: float = 0.40
    transformer: float = 0.50
    rule: float = 0.10


def build_rule_based_proba(rule_features: np.ndarray) -> np.ndarray:
    overlap_ctx_resp = rule_features[:, 0]
    overlap_prompt_resp = rule_features[:, 1]
    neg_mismatch = rule_features[:, 2]
    extrinsic_hint = rule_features[:, 3]

    no_score = (0.55 * overlap_ctx_resp + 0.25 * overlap_prompt_resp) * (1.0 - 0.4 * neg_mismatch)
    intrinsic_score = 0.6 * neg_mismatch + 0.2 * (1.0 - overlap_ctx_resp)
    extrinsic_score = 0.6 * extrinsic_hint + 0.3 * (1.0 - overlap_prompt_resp)

    raw = np.stack([no_score, intrinsic_score, extrinsic_score], axis=1)
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum(axis=1, keepdims=True)


class HybridEnsembler:
    def __init__(self, weights: HybridWeights | None = None) -> None:
        self.weights = weights or HybridWeights()

    def fit(
        self,
        y_true: np.ndarray,
        proba_classical: np.ndarray,
        proba_transformer: np.ndarray,
        proba_rule: np.ndarray,
    ) -> None:
        def objective(weight_vec: np.ndarray) -> float:
            w = np.clip(weight_vec, 0.0, 1.0)
            if np.sum(w) == 0:
                w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
            else:
                w = w / np.sum(w)

            blended = w[0] * proba_classical + w[1] * proba_transformer + w[2] * proba_rule
            preds = np.argmax(blended, axis=1)
            score = f1_score(y_true, preds, average="macro")
            return -score

        x0 = np.array([self.weights.classical, self.weights.transformer, self.weights.rule], dtype=np.float64)
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)

        best = result.x if result.success else x0
        best = np.clip(best, 0.0, 1.0)
        best = best / np.sum(best)

        self.weights = HybridWeights(
            classical=float(best[0]),
            transformer=float(best[1]),
            rule=float(best[2]),
        )

    def predict_proba(
        self,
        proba_classical: np.ndarray,
        proba_transformer: np.ndarray,
        proba_rule: np.ndarray,
    ) -> np.ndarray:
        w = self.weights
        return w.classical * proba_classical + w.transformer * proba_transformer + w.rule * proba_rule

    def predict(
        self,
        proba_classical: np.ndarray,
        proba_transformer: np.ndarray,
        proba_rule: np.ndarray,
    ) -> np.ndarray:
        return np.argmax(self.predict_proba(proba_classical, proba_transformer, proba_rule), axis=1)
