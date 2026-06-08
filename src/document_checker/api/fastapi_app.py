from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.document_checker.models.text_classifier import load_text_classifier
from src.document_checker.pipeline.document_pipeline import (
    DocumentPredictionResult,
    predict_document_from_image_result,
)


app = FastAPI(
    title="Document Checker API",
    version="0.1.0",
    description="CPU-only prototype: document -> OCR -> text -> ML -> result",
)


@app.get("/health")
def health() -> dict[str, str]:
    """
    Простой health-check: проверяем, что модель классификатора доступна.
    """
    try:
        load_text_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Model file not found: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(description="Image file (tif/png/jpg)")],
    lang: str = "eng",
) -> JSONResponse:
    """
    Принимает одно изображение, запускает OCR + классификацию и возвращает результат.
    """
    content_type = file.content_type or ""
    filename = file.filename or "uploaded"
    suffix = Path(filename).suffix.lower()

    allowed_suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    is_image_content_type = content_type.startswith("image/")
    is_allowed_suffix = suffix in allowed_suffixes

    if not (is_image_content_type or is_allowed_suffix):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: content_type={content_type or 'unknown'}, "
                f"filename={filename}. "
                f"Expected image upload with one of extensions: "
                f"{', '.join(sorted(allowed_suffixes))}."
            ),
        )

    try:
        temp_dir = Path("/tmp/document_checker_api")
        temp_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = Path(filename).stem or "image"
        temp_path = temp_dir / f"upload_{safe_stem}{suffix or '.tif'}"

        with temp_path.open("wb") as f:
            f.write(await file.read())

        result: DocumentPredictionResult = predict_document_from_image_result(
            image_path=temp_path,
            lang=lang,
        )

        return JSONResponse(content=result.to_dict())

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc