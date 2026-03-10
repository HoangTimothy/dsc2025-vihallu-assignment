import re
import unicodedata


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text


def serialize_triplet(context: str, prompt: str, response: str) -> str:
    context = normalize_text(context)
    prompt = normalize_text(prompt)
    response = normalize_text(response)
    return f"[CLS] {context} [SEP] {prompt} [SEP] {response} [SEP]"


def simple_tokenize(text: str) -> list[str]:
    text = normalize_text(text).lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)
