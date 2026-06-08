from __future__ import annotations

import pytest

from src.document_checker.models.text_classifier import predict_document_label


def test_predict_document_label_raises_on_empty_text() -> None:
    with pytest.raises(ValueError, match="Input text for classification is empty."):
        predict_document_label("")