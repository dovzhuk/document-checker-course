from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "text_tfidf_linearsvc.joblib"


def load_text_classifier(model_path: Path | None = None):
    path = model_path or DEFAULT_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def predict_document_label(text: str, model_path: Path | None = None) -> str:
    model = load_text_classifier(model_path)
    prediction = model.predict([text])
    return str(prediction[0])


def predict_document_labels(texts: Iterable[str], model_path: Path | None = None) -> list[str]:
    model = load_text_classifier(model_path)
    predictions = model.predict(list(texts))
    return [str(label) for label in predictions]
