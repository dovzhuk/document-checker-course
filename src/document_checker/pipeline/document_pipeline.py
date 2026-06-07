from __future__ import annotations

from pathlib import Path

from src.document_checker.models.text_classifier import predict_document_label
from src.document_checker.ocr.tesseract import run_tesseract_on_image_psm6


def predict_document_from_image(
    image_path: Path | str,
    lang: str = "eng",
) -> dict[str, str]:
    path = Path(image_path)

    ocr_text = run_tesseract_on_image_psm6(path, lang=lang)
    predicted_label = predict_document_label(ocr_text)

    return {
        "image_path": str(path),
        "ocr_text": ocr_text,
        "predicted_label": predicted_label,
    }