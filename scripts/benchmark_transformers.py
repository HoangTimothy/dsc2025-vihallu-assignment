from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vihallu.data import read_train_csv, stratified_sample, stratified_split
from vihallu.metrics import build_report, compute_scores
from vihallu.models.transformer import TransformerConfig, TransformerHallucinationModel, resolve_text_mode
from vihallu.utils.io_utils import ensure_dir, save_json


DEFAULT_MODELS = ["vinai/phobert-base-v2", "xlm-roberta-base"]
DEFAULT_MAX_LENGTHS = [256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/transformer_benchmark")
    parser.add_argument("--model_names", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max_lengths", nargs="+", type=int, default=DEFAULT_MAX_LENGTHS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--text_mode", choices=["auto", "raw", "word_segmented"], default="auto")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def run_name(model_name: str, max_length: int) -> str:
    model_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return f"{model_part}_len{max_length}"


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    save_json({"runs": rows}, output_dir / "summary.json")


def evaluate_and_save(
    model: TransformerHallucinationModel,
    trainer,
    valid_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    eval_result = trainer.predict(model._tokenize(model._to_dataset(valid_df, with_label=True)))
    valid_true = valid_df["label_id"].to_numpy()
    valid_pred = eval_result.predictions.argmax(axis=1)

    scores = compute_scores(valid_true, valid_pred)
    report = build_report(valid_true, valid_pred)

    logits = eval_result.predictions
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    np.save(output_dir / "valid_ids.npy", valid_df["id"].to_numpy())
    np.save(output_dir / "valid_true.npy", valid_true)
    np.save(output_dir / "valid_proba.npy", proba)
    save_json(scores, output_dir / "scores.json")
    save_json(report, output_dir / "report.json")
    return scores


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows: list[dict[str, Any]] = []

    for model_name in args.model_names:
        text_mode = resolve_text_mode(model_name, args.text_mode)
        df = read_train_csv(args.train_path, text_mode=text_mode)
        df = stratified_sample(df, sample_size=args.sample_size, seed=args.seed)
        bundle = stratified_split(df, valid_size=args.valid_size, seed=args.seed)

        for max_length in args.max_lengths:
            current_run_name = run_name(model_name, max_length)
            run_dir = ensure_dir(output_dir / current_run_name)
            config = TransformerConfig(
                model_name=model_name,
                max_length=max_length,
                text_mode=text_mode,
                learning_rate=args.learning_rate,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                epochs=args.epochs,
                weight_decay=args.weight_decay,
                seed=args.seed,
                fp16=args.fp16,
                bf16=args.bf16,
            )

            started = time.perf_counter()
            row: dict[str, Any] = {
                "run_name": current_run_name,
                "model_name": model_name,
                "max_length": max_length,
                "text_mode": text_mode,
                "output_dir": str(run_dir),
                "status": "running",
            }
            rows.append(row)
            write_summary(rows, output_dir)

            try:
                print(f"[benchmark] start {current_run_name}")
                model = TransformerHallucinationModel(config)
                trainer = model.fit(bundle.train, bundle.valid, output_dir=str(run_dir))
                scores = evaluate_and_save(model, trainer, bundle.valid, run_dir)

                elapsed_minutes = (time.perf_counter() - started) / 60.0
                metadata = {
                    **asdict(config),
                    "train_rows": int(len(bundle.train)),
                    "valid_rows": int(len(bundle.valid)),
                    "valid_size": float(args.valid_size),
                    "elapsed_minutes": elapsed_minutes,
                }
                save_json(metadata, run_dir / "model_metadata.json")

                row.update(
                    {
                        "status": "done",
                        "macro_f1": scores["macro_f1"],
                        "accuracy": scores["accuracy"],
                        "elapsed_minutes": elapsed_minutes,
                    }
                )
                print(f"[benchmark] done {current_run_name} {scores}")
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "error": repr(exc),
                        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
                    }
                )
                print(f"[benchmark] failed {current_run_name}: {exc}")
                if args.fail_fast:
                    write_summary(rows, output_dir)
                    raise
            finally:
                write_summary(rows, output_dir)

    summary_df = pd.DataFrame(rows)
    done_df = summary_df[summary_df["status"] == "done"].copy()
    if not done_df.empty:
        done_df = done_df.sort_values("macro_f1", ascending=False)
        print(done_df[["run_name", "macro_f1", "accuracy", "elapsed_minutes"]].to_string(index=False))


if __name__ == "__main__":
    main()
