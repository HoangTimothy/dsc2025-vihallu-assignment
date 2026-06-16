from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ID_COLUMN, LABEL2ID, TEST_LABEL_COLUMN, TEXT_COLUMNS, TRAIN_LABEL_COLUMN
from .utils.text_utils import serialize_triplet


@dataclass
class DataBundle:
    train: pd.DataFrame
    valid: pd.DataFrame


def resolve_data_path(path: str | Path) -> Path:
    input_path = Path(path)
    if input_path.exists():
        return input_path

    tried = [input_path]
    if not input_path.is_absolute():
        kaggle_input = Path("/kaggle/input")
        candidates = [
            kaggle_input / "vihallu" / input_path.name,
            kaggle_input / input_path.name,
        ]
        if kaggle_input.exists():
            candidates.extend(sorted(kaggle_input.rglob(input_path.name)))

        for candidate in candidates:
            if candidate not in tried:
                tried.append(candidate)
            if candidate.exists():
                return candidate

    tried_text = "\n".join(f"  - {candidate}" for candidate in tried)
    raise FileNotFoundError(f"Could not find data file '{path}'. Tried:\n{tried_text}")


def read_train_csv(path: str | Path, text_mode: str = "raw") -> pd.DataFrame:
    df = pd.read_csv(resolve_data_path(path))
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
        lambda row: serialize_triplet(row["context"], row["prompt"], row["response"], text_mode=text_mode),
        axis=1,
    )
    return df


def read_test_csv(path: str | Path, text_mode: str = "raw") -> pd.DataFrame:
    df = pd.read_csv(resolve_data_path(path))
    required = [ID_COLUMN, *TEXT_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in test csv: {missing}")

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    if TEST_LABEL_COLUMN not in df.columns:
        df[TEST_LABEL_COLUMN] = ""

    df["serialized_text"] = df.apply(
        lambda row: serialize_triplet(row["context"], row["prompt"], row["response"], text_mode=text_mode),
        axis=1,
    )
    return df


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int = 42) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True)

    num_labels = df["label_id"].nunique()
    if sample_size < num_labels:
        raise ValueError(f"sample_size must be at least {num_labels} for stratified sampling.")

    _, sample_df = train_test_split(
        df,
        test_size=sample_size,
        random_state=seed,
        stratify=df["label_id"],
    )
    return sample_df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, valid_size: float = 0.15, seed: int = 42) -> DataBundle:
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        stratify=df["label_id"],
    )
    return DataBundle(train=train_df.reset_index(drop=True), valid=valid_df.reset_index(drop=True))
