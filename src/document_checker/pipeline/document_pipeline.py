from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.document_checker.models.text_classifier import predict_document_label
from src.document_checker.ocr.tesseract import run_tesseract_on_image_psm6


@dataclass(slots=True)
class DocumentPredictionResult:
    image_path: str
    ocr_text: str
    predicted_label: str
    lang: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def predict_document_from_image(
    image_path: Path | str,
    lang: str = "eng",
) -> dict[str, str]:
    return predict_document_from_image_result(image_path=image_path, lang=lang).to_dict()


def predict_document_from_image_result(
    image_path: Path | str,
    lang: str = "eng",
) -> DocumentPredictionResult:
    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    ocr_text = run_tesseract_on_image_psm6(path, lang=lang).strip()

    if not ocr_text:
        raise ValueError(f"OCR returned empty text for file: {path}")

    predicted_label = predict_document_label(ocr_text)

    return DocumentPredictionResult(
        image_path=str(path),
        ocr_text=ocr_text,
        predicted_label=predicted_label,
        lang=lang,
    )