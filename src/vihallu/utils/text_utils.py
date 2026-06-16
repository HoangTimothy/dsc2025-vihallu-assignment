import re
import unicodedata
from functools import lru_cache
from typing import Callable


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text


@lru_cache(maxsize=1)
def _get_word_segmenter() -> Callable[[str], str] | None:
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize
    except Exception:
        pass

    try:
        from underthesea import word_tokenize

        return lambda text: word_tokenize(text, format="text")
    except Exception:
        return None


def word_segment_text(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    segmenter = _get_word_segmenter()
    if segmenter is None:
        return text
    return segmenter(text)


def serialize_triplet(context: str, prompt: str, response: str, text_mode: str = "raw") -> str:
    normalizer = word_segment_text if text_mode == "word_segmented" else normalize_text
    context = normalizer(context)
    prompt = normalizer(prompt)
    response = normalizer(response)
    return f"[CLS] {context} [SEP] {prompt} [SEP] {response} [SEP]"


def simple_tokenize(text: str) -> list[str]:
    text = normalize_text(text).lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)
