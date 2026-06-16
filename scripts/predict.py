from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

os.environ.setdefault("HF_HOME", str(ROOT_DIR / "outputs" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from vihallu.config import ID2LABEL
from vihallu.data import read_test_csv
from vihallu.features import build_hybrid_features
from vihallu.models.hybrid import HybridWeights, build_rule_based_proba
from vihallu.utils.io_utils import load_joblib, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["classical", "transformer", "hybrid"], required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--classical_dir", type=str, default="outputs/classical")
    parser.add_argument("--transformer_dir", type=str, default="outputs/transformer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=None)
    return parser.parse_args()


def load_transformer_metadata(model_dir: str) -> dict:
    metadata_path = Path(model_dir) / "model_metadata.json"
    if metadata_path.exists():
        return load_json(metadata_path)
    return {"max_length": 256, "text_mode": "raw"}


def predict_with_transformer(df, model_dir: str, batch_size: int, max_length: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    tokens = tokenizer(
        df["serialized_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    class TinyDataset:
        def __len__(self):
            return tokens["input_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in tokens.items()}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="tmp_predict", per_device_eval_batch_size=batch_size, report_to="none"),
    )

    outputs = trainer.predict(TinyDataset())
    logits = outputs.predictions
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    df = read_test_csv(args.test_path, text_mode="raw")

    if args.model_type == "classical":
        model = load_joblib(Path(args.model_dir) / "classical_model.joblib")
        pred = model.predict(df["serialized_text"].tolist())

    elif args.model_type == "transformer":
        metadata = load_transformer_metadata(args.model_dir)
        text_mode = str(metadata.get("text_mode", "raw"))
        max_length = args.max_length or int(metadata.get("max_length", 256))
        transformer_df = df if text_mode == "raw" else read_test_csv(args.test_path, text_mode=text_mode)
        proba = predict_with_transformer(transformer_df, args.model_dir, args.batch_size, max_length)
        pred = np.argmax(proba, axis=1)

    else:
        classical_model = load_joblib(Path(args.classical_dir) / "classical_model.joblib")
        proba_classical = classical_model.predict_proba(df["serialized_text"].tolist())

        metadata = load_transformer_metadata(args.transformer_dir)
        text_mode = str(metadata.get("text_mode", "raw"))
        max_length = args.max_length or int(metadata.get("max_length", 256))
        transformer_df = df if text_mode == "raw" else read_test_csv(args.test_path, text_mode=text_mode)
        proba_transformer = predict_with_transformer(
            transformer_df,
            args.transformer_dir,
            args.batch_size,
            max_length,
        )
        proba_rule = build_rule_based_proba(build_hybrid_features(df))

        weights_dict = load_json(Path(args.model_dir) / "hybrid_weights.json")
        weights = HybridWeights(
            classical=float(weights_dict["classical"]),
            transformer=float(weights_dict["transformer"]),
            rule=float(weights_dict["rule"]),
        )

        blended = (
            weights.classical * proba_classical
            + weights.transformer * proba_transformer
            + weights.rule * proba_rule
        )
        pred = np.argmax(blended, axis=1)

    df["predict_label"] = [ID2LABEL[int(i)] for i in pred]
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    df[["id", "context", "prompt", "response", "predict_label"]].to_csv(args.output_path, index=False)

    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
