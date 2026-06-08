from __future__ import annotations

from fastapi.testclient import TestClient

from src.document_checker.api.fastapi_app import app


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_mocked_result(monkeypatch) -> None:
    from src.document_checker.pipeline.document_pipeline import DocumentPredictionResult

    def mock_predict_document_from_image_result(image_path, lang="eng"):
        return DocumentPredictionResult(
            image_path=str(image_path),
            ocr_text="mock ocr text",
            predicted_label="file_folder",
            lang=lang,
        )

    monkeypatch.setattr(
        "src.document_checker.api.fastapi_app.predict_document_from_image_result",
        mock_predict_document_from_image_result,
    )

    files = {
        "file": ("sample.tif", b"fake-image-bytes", "application/octet-stream"),
    }
    data = {"lang": "eng"}

    response = client.post("/predict", files=files, data=data)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "file_folder"
    assert body["ocr_text"] == "mock ocr text"
    assert body["lang"] == "eng"