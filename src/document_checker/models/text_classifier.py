from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "text_tfidf_linearsvc.joblib"


@lru_cache(maxsize=4)
def _load_text_classifier_cached(model_path_str: str):
    path = Path(model_path_str)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def load_text_classifier(model_path: Path | None = None):
    path = (model_path or DEFAULT_MODEL_PATH).resolve()
    return _load_text_classifier_cached(str(path))


def predict_document_label(text: str, model_path: Path | None = None) -> str:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Input text for classification is empty.")

    model = load_text_classifier(model_path)
    prediction = model.predict([normalized_text])
    return str(prediction[0])


def predict_document_labels(texts: Iterable[str], model_path: Path | None = None) -> list[str]:
    text_list = [str(text).strip() for text in texts]

    if not text_list:
        raise ValueError("Input texts for classification are empty.")

    if any(not text for text in text_list):
        raise ValueError("One or more input texts for classification are empty.")

    model = load_text_classifier(model_path)
    predictions = model.predict(text_list)
    return [str(label) for label in predictions]