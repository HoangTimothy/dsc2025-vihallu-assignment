from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class ClassicalHallucinationModel:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=120000,
                        lowercase=False,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        C=2.0,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: np.ndarray) -> "ClassicalHallucinationModel":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)
