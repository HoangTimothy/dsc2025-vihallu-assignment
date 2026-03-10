from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ID_COLUMN, LABEL2ID, TEST_LABEL_COLUMN, TEXT_COLUMNS, TRAIN_LABEL_COLUMN
from .utils.text_utils import serialize_triplet


@dataclass
class DataBundle:
    train: pd.DataFrame
    valid: pd.DataFrame


def read_train_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [ID_COLUMN, *TEXT_COLUMNS, TRAIN_LABEL_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train csv: {missing}")

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    df[TRAIN_LABEL_COLUMN] = df[TRAIN_LABEL_COLUMN].astype(str)
    unknown = sorted(set(df[TRAIN_LABEL_COLUMN]) - set(LABEL2ID))
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")

    df["label_id"] = df[TRAIN_LABEL_COLUMN].map(LABEL2ID)
    df["serialized_text"] = df.apply(
        lambda row: serialize_triplet(row["context"], row["prompt"], row["response"]), axis=1
    )
    return df


def read_test_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [ID_COLUMN, *TEXT_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in test csv: {missing}")

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    if TEST_LABEL_COLUMN not in df.columns:
        df[TEST_LABEL_COLUMN] = ""

    df["serialized_text"] = df.apply(
        lambda row: serialize_triplet(row["context"], row["prompt"], row["response"]), axis=1
    )
    return df


def stratified_split(df: pd.DataFrame, valid_size: float = 0.15, seed: int = 42) -> DataBundle:
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        stratify=df["label_id"],
    )
    return DataBundle(train=train_df.reset_index(drop=True), valid=valid_df.reset_index(drop=True))
