from __future__ import annotations

import numpy as np
import pandas as pd

from .utils.text_utils import simple_tokenize

NEGATION_CUES = {"không", "chưa", "chẳng", "khỏi", "chả"}
EXTRINSIC_HINTS = {
    "ngoài ra",
    "bên cạnh đó",
    "có thể",
    "dường như",
    "nhiều nguồn",
    "theo một số",
}


def _token_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / max(union, 1)


def _contains_any(text: str, cues: set[str]) -> int:
    text = text.lower()
    return int(any(cue in text for cue in cues))


def build_hybrid_features(df: pd.DataFrame) -> np.ndarray:
    rows: list[list[float]] = []

    for _, row in df.iterrows():
        context = str(row["context"])
        prompt = str(row["prompt"])
        response = str(row["response"])

        context_toks = simple_tokenize(context)
        prompt_toks = simple_tokenize(prompt)
        response_toks = simple_tokenize(response)

        overlap_ctx_resp = _token_overlap(context_toks, response_toks)
        overlap_prompt_resp = _token_overlap(prompt_toks, response_toks)

        neg_ctx = _contains_any(context, NEGATION_CUES)
        neg_resp = _contains_any(response, NEGATION_CUES)
        neg_mismatch = int(neg_ctx != neg_resp)

        extrinsic_hint = _contains_any(response, EXTRINSIC_HINTS)

        len_ctx = max(len(context_toks), 1)
        len_resp = len(response_toks)
        response_length_ratio = len_resp / len_ctx

        rows.append(
            [
                overlap_ctx_resp,
                overlap_prompt_resp,
                float(neg_mismatch),
                float(extrinsic_hint),
                response_length_ratio,
            ]
        )

    return np.array(rows, dtype=np.float32)
