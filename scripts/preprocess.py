"""Preprocess `vihallu-train.csv` for Vietnamese NLP.

Creates cleaned and tokenized columns for `context`, `prompt`, and `response`.
Saves output to `outputs/eda/preprocessed_train.csv` by default.

Usage:
    python scripts/preprocess.py --input vihallu-train.csv --output outputs/eda/preprocessed_train.csv

This script uses `pyvi` (ViTokenizer) when available for tokenization, falls back to
`underthesea.word_tokenize` if not. It performs basic normalization and cleaning.
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata
import json
from typing import Optional

import pandas as pd

try:
    from pyvi import ViTokenizer
    _HAS_PYVI = True
except Exception:
    _HAS_PYVI = False

try:
    from underthesea import word_tokenize as uts_word_tokenize
    _HAS_UTS = True
except Exception:
    _HAS_UTS = False


RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\S+@\S+\.(?:com|net|org|edu|vn|io|gov|info)")
RE_HTML = re.compile(r"<[^>]+>")
RE_MULTI_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    s = s.replace('\r', ' ').replace('\n', ' ')
    s = RE_HTML.sub(' ', s)
    s = RE_URL.sub(' ', s)
    s = RE_EMAIL.sub(' ', s)
    s = s.strip()
    # Unicode normalize (keep accents)
    s = unicodedata.normalize('NFC', s)
    # collapse spaces
    s = RE_MULTI_WS.sub(' ', s)
    return s


def lower_text(text: str) -> str:
    try:
        return text.lower()
    except Exception:
        return text


def tokenize_text(text: str) -> str:
    if not text:
        return ""
    if _HAS_PYVI:
        try:
            return ViTokenizer.tokenize(text)
        except Exception:
            pass
    if _HAS_UTS:
        try:
            return uts_word_tokenize(text, format='text')
        except Exception:
            pass
    # naive fallback: whitespace tokenization
    return ' '.join(text.split())


def preprocess_df(df: pd.DataFrame, text_cols: list[str], method: str = 'baseline') -> pd.DataFrame:
    for col in text_cols:
        clean_col = f"{col}_clean"
        # Làm sạch: HTML, URL, normalize NFC, etc. is done in normalize_text
        df[clean_col] = df[col].fillna("").astype(str).map(normalize_text)
        
        # Tokenization & Formatting based on method
        if method in ['phobert', 'baseline', 'hybrid']:
            # PhoBERT và các model truyền thống thường cần word segmentation với underscore
            df[f"{col}_tok"] = df[clean_col].map(lower_text).map(tokenize_text)
            df[f"{col}_char_len"] = df[clean_col].map(len)
            df[f"{col}_token_len"] = df[f"{col}_tok"].map(lambda x: len(x.split()) if x else 0)
        else:
            # LLMs (Qwen, xlm-roberta) thường dùng subword tokenizers eigene của chúng, không cần split tay bằng underscore.
            df[f"{col}_tok"] = df[clean_col]
            df[f"{col}_char_len"] = df[clean_col].map(len)
            # Tạm tính token bằng khoảng trắng
            df[f"{col}_token_len"] = df[clean_col].map(lambda x: len(x.split()) if x else 0)

    # Prompt Formatting cho LLM Fine-tune
    if method in ['llm', 'semantic_entropy', 'mhad', 'halu_agent']:
        # Format template rõ ràng cho LLM
        def build_llm_prompt(row):
            return f"Context: {row.get('context_clean', '')}\nQuestion: {row.get('prompt_clean', '')}\nModel Answer: {row.get('response_clean', '')}\nIs the model answer hallucinated based on the context? (Yes/No):"
        df["llm_prompt"] = df.apply(build_llm_prompt, axis=1)

    return df


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='vihallu-train.csv')
    parser.add_argument('--output', '-o', default='outputs/eda/preprocessed_train.csv')
    parser.add_argument('--sample', '-s', type=int, default=0, help='Optional sample size to process')
    parser.add_argument('--method', '-m', default='baseline', 
                        choices=['baseline', 'phobert', 'xlm-roberta', 'llm', 'semantic_entropy', 'mhad', 'halu_agent', 'hybrid'],
                        help='Preprocessing method based on the target model architecture')
    parsed = parser.parse_args() if args is None else args

    inp = parsed.input
    out = parsed.output
    method = parsed.method
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print('Loading', inp)
    df = pd.read_csv(inp)
    if parsed.sample and parsed.sample > 0:
        df = df.sample(parsed.sample, random_state=0).reset_index(drop=True)

    text_cols = [c for c in ['context', 'prompt', 'response'] if c in df.columns]
    if not text_cols:
        raise SystemExit('No text columns found in input CSV (expected context/prompt/response)')

    print(f'Using tokenizers: pyvi={_HAS_PYVI}, underthesea={_HAS_UTS}')
    print(f'Applying preprocessing method: {method}')
    df = preprocess_df(df, text_cols, method=method)

    # Basic stats
    stats = {
        'rows': int(len(df)),
        'columns': df.shape[1],
        'sample_rows': df.head(3).to_dict(orient='records')
    }
    with open(os.path.join(os.path.dirname(out), 'preprocess_stats.json'), 'w', encoding='utf8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    df.to_csv(out, index=False)
    print('Saved preprocessed CSV to', out)


if __name__ == '__main__':
    main()
