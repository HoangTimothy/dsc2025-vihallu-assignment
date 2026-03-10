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
from vihallu.metrics import build_report, compute_scores
from vihallu.models.transformer import TransformerConfig, TransformerHallucinationModel
from vihallu.utils.io_utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/transformer")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = read_train_csv(args.train_path)
    bundle = stratified_split(df, valid_size=args.valid_size, seed=args.seed)

    config = TransformerConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
    )

    model = TransformerHallucinationModel(config)
    trainer = model.fit(bundle.train, bundle.valid, output_dir=str(output_dir))

    eval_result = trainer.predict(model._tokenize(model._to_dataset(bundle.valid, with_label=True)))
    valid_true = bundle.valid["label_id"].to_numpy()
    valid_pred = eval_result.predictions.argmax(axis=1)

    scores = compute_scores(valid_true, valid_pred)
    report = build_report(valid_true, valid_pred)

    np.save(Path(output_dir) / "valid_ids.npy", bundle.valid["id"].to_numpy())
    np.save(Path(output_dir) / "valid_true.npy", valid_true)

    logits = eval_result.predictions
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    np.save(Path(output_dir) / "valid_proba.npy", proba)

    save_json(scores, Path(output_dir) / "scores.json")
    save_json(report, Path(output_dir) / "report.json")

    print("[transformer] done", scores)


if __name__ == "__main__":
    main()
