from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vihallu.utils.io_utils import ensure_dir, load_json, save_json


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
    parser.add_argument("--skip_done", action="store_true")
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


def resolve_text_mode(model_name: str, text_mode: str = "auto") -> str:
    if text_mode != "auto":
        return text_mode
    return "word_segmented" if "phobert" in model_name.lower() else "raw"


def build_train_command(args: argparse.Namespace, model_name: str, max_length: int, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "train_transformer.py"),
        "--train_path",
        args.train_path,
        "--output_dir",
        str(run_dir),
        "--model_name",
        model_name,
        "--max_length",
        str(max_length),
        "--epochs",
        str(args.epochs),
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--valid_size",
        str(args.valid_size),
        "--seed",
        str(args.seed),
        "--sample_size",
        str(args.sample_size),
        "--text_mode",
        args.text_mode,
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")
    return cmd


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows: list[dict[str, Any]] = []

    for model_name in args.model_names:
        text_mode = resolve_text_mode(model_name, args.text_mode)

        for max_length in args.max_lengths:
            current_run_name = run_name(model_name, max_length)
            run_dir = ensure_dir(output_dir / current_run_name)
            scores_path = run_dir / "scores.json"

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
                if args.skip_done and scores_path.exists():
                    scores = load_json(scores_path)
                    metadata_path = run_dir / "model_metadata.json"
                    metadata = load_json(metadata_path) if metadata_path.exists() else {}
                    row.update(
                        {
                            "status": "skipped",
                            "macro_f1": scores["macro_f1"],
                            "accuracy": scores["accuracy"],
                            "elapsed_minutes": metadata.get("elapsed_minutes", 0.0),
                        }
                    )
                    print(f"[benchmark] skip {current_run_name} {scores}")
                    continue

                print(f"[benchmark] start {current_run_name}")
                subprocess.run(build_train_command(args, model_name, max_length, run_dir), check=True)
                scores = load_json(scores_path)
                elapsed_minutes = (time.perf_counter() - started) / 60.0

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
    done_df = summary_df[summary_df["status"].isin(["done", "skipped"])].copy()
    if not done_df.empty:
        done_df = done_df.sort_values("macro_f1", ascending=False)
        print(done_df[["run_name", "macro_f1", "accuracy", "elapsed_minutes"]].to_string(index=False))


if __name__ == "__main__":
    main()
