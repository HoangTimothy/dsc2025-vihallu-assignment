from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vihallu.data import read_train_csv, stratified_split
from vihallu.features import build_hybrid_features
from vihallu.metrics import build_report, compute_scores
from vihallu.models.hybrid import HybridEnsembler, HybridWeights, build_rule_based_proba
from vihallu.utils.io_utils import ensure_dir, load_joblib, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--classical_dir", type=str, required=True)
    parser.add_argument("--transformer_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/hybrid")
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = read_train_csv(args.train_path)
    bundle = stratified_split(df, valid_size=args.valid_size, seed=args.seed)

    classical_model = load_joblib(Path(args.classical_dir) / "classical_model.joblib")
    proba_classical = classical_model.predict_proba(bundle.valid["serialized_text"].tolist())

    trans_valid_proba_path = Path(args.transformer_dir) / "valid_proba.npy"
    if not trans_valid_proba_path.exists():
        raise FileNotFoundError(
            f"Missing transformer validation probabilities: {trans_valid_proba_path}. Run train_transformer first."
        )
    proba_transformer = np.load(trans_valid_proba_path)

    rule_features = build_hybrid_features(bundle.valid)
    proba_rule = build_rule_based_proba(rule_features)

    valid_true = bundle.valid["label_id"].to_numpy()

    ensembler = HybridEnsembler(weights=HybridWeights(0.40, 0.50, 0.10))
    ensembler.fit(valid_true, proba_classical, proba_transformer, proba_rule)

    valid_pred = ensembler.predict(proba_classical, proba_transformer, proba_rule)
    scores = compute_scores(valid_true, valid_pred)
    report = build_report(valid_true, valid_pred)

    weights_payload = {
        "classical": ensembler.weights.classical,
        "transformer": ensembler.weights.transformer,
        "rule": ensembler.weights.rule,
    }

    np.save(Path(output_dir) / "valid_ids.npy", bundle.valid["id"].to_numpy())
    np.save(Path(output_dir) / "valid_true.npy", valid_true)
    np.save(Path(output_dir) / "valid_pred.npy", valid_pred)

    save_json(weights_payload, Path(output_dir) / "hybrid_weights.json")
    save_json(scores, Path(output_dir) / "scores.json")
    save_json(report, Path(output_dir) / "report.json")

    print("[hybrid] done", scores)
    print("[hybrid] weights", weights_payload)


if __name__ == "__main__":
    main()
