#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.document_checker.preprocessing.text_cleaning import SimpleTextCleaner


DATASET_CSV = PROJECT_ROOT / "data" / "ocr_rvl_cdip" / "ocr_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
MODEL_PATH = MODEL_DIR / "text_tfidf_linearsvc.joblib"


def main() -> None:
    df = pd.read_csv(DATASET_CSV)

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("clean", SimpleTextCleaner()),
            ("tfidf", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                lowercase=False,
            )),
            ("clf", LinearSVC()),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print(f"Validation accuracy (LinearSVC): {acc:.4f}")
    print(f"Validation macro F1 (LinearSVC): {f1_macro:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()