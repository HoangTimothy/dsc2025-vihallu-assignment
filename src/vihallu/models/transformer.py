from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(ROOT_DIR / "outputs" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..config import ID2LABEL, LABEL2ID


@dataclass
class TransformerConfig:
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    text_mode: str = "raw"
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    epochs: int = 3
    weight_decay: float = 0.01
    seed: int = 42
    fp16: bool = False
    bf16: bool = False


def resolve_text_mode(model_name: str, text_mode: str = "auto") -> str:
    if text_mode != "auto":
        return text_mode
    return "word_segmented" if "phobert" in model_name.lower() else "raw"


class TransformerHallucinationModel:
    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    def _to_dataset(self, df: pd.DataFrame, with_label: bool = True) -> Dataset:
        payload = {
            "text": df["serialized_text"].tolist(),
        }
        if with_label:
            payload["labels"] = df["label_id"].tolist()
        return Dataset.from_dict(payload)

    def _tokenize(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            lambda batch: self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_length,
            ),
            batched=True,
        )

    @staticmethod
    def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": float(macro_f1)}

    def _trainer_tokenizer_kwargs(self) -> dict[str, object]:
        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_params:
            return {"processing_class": self.tokenizer}
        if "tokenizer" in trainer_params:
            return {"tokenizer": self.tokenizer}
        return {}

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, output_dir: str) -> Trainer:
        train_ds = self._tokenize(self._to_dataset(train_df, with_label=True))
        valid_ds = self._tokenize(self._to_dataset(valid_df, with_label=True))

        training_kwargs = {
            "output_dir": output_dir,
            "learning_rate": self.config.learning_rate,
            "per_device_train_batch_size": self.config.train_batch_size,
            "per_device_eval_batch_size": self.config.eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_train_epochs": self.config.epochs,
            "weight_decay": self.config.weight_decay,
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "greater_is_better": True,
            "logging_strategy": "steps",
            "logging_steps": 50,
            "report_to": "none",
            "save_total_limit": 1,
            "seed": self.config.seed,
            "data_seed": self.config.seed,
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
        }
        strategy_arg = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
        training_kwargs[strategy_arg] = "epoch"
        args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
            **self._trainer_tokenizer_kwargs(),
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def predict_proba(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        dataset = self._tokenize(self._to_dataset(df, with_label=False))
        args = TrainingArguments(
            output_dir="tmp_predict",
            per_device_eval_batch_size=batch_size,
            report_to="none",
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            **self._trainer_tokenizer_kwargs(),
        )
        outputs = trainer.predict(dataset)
        logits = outputs.predictions
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)
