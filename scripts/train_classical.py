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
from vihallu.models.classical import ClassicalHallucinationModel
from vihallu.utils.io_utils import ensure_dir, save_joblib, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/classical")
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = read_train_csv(args.train_path)
    bundle = stratified_split(df, valid_size=args.valid_size, seed=args.seed)

    model = ClassicalHallucinationModel().fit(
        bundle.train["serialized_text"].tolist(),
        bundle.train["label_id"].to_numpy(),
    )

    valid_pred = model.predict(bundle.valid["serialized_text"].tolist())
    valid_true = bundle.valid["label_id"].to_numpy()

    scores = compute_scores(valid_true, valid_pred)
    report = build_report(valid_true, valid_pred)

    save_joblib(model, Path(output_dir) / "classical_model.joblib")
    save_json(scores, Path(output_dir) / "scores.json")
    save_json(report, Path(output_dir) / "report.json")

    np.save(Path(output_dir) / "valid_ids.npy", bundle.valid["id"].to_numpy())
    np.save(Path(output_dir) / "valid_true.npy", valid_true)
    np.save(
        Path(output_dir) / "valid_proba.npy",
        model.predict_proba(bundle.valid["serialized_text"].tolist()),
    )

    print("[classical] done", scores)


if __name__ == "__main__":
    main()
