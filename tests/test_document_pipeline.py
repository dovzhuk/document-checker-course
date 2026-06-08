from __future__ import annotations

from pathlib import Path

import pytest

from src.document_checker.pipeline.document_pipeline import predict_document_from_image_result


def test_predict_document_from_image_result_raises_for_missing_file() -> None:
    missing_path = Path("does_not_exist_12345.tif")

    with pytest.raises(FileNotFoundError, match="Image file not found:"):
        predict_document_from_image_result(missing_path)