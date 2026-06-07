from __future__ import annotations

import re

from sklearn.base import BaseEstimator, TransformerMixin


class SimpleTextCleaner(BaseEstimator, TransformerMixin):
    """Простая очистка текста: lower, убираем лишние пробелы и управляющие символы."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for text in X:
            t = str(text)
            t = t.lower()
            t = re.sub(r"\s+", " ", t)
            t = t.strip()
            cleaned.append(t)
        return cleaned