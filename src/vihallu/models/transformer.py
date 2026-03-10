from __future__ import annotations

from dataclasses import dataclass

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
    model_name: str = "vinai/phobert-base"
    max_length: int = 256
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.01


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

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, output_dir: str) -> Trainer:
        train_ds = self._tokenize(self._to_dataset(train_df, with_label=True))
        valid_ds = self._tokenize(self._to_dataset(valid_df, with_label=True))

        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
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
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )
        outputs = trainer.predict(dataset)
        logits = outputs.predictions
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)
