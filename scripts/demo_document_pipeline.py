#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для запуска как скрипта
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_checker.pipeline.document_pipeline import predict_document_from_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR + text classification on a single document image."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the document image (e.g. .tif from data/ocr_rvl_cdip/images_tif/...)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Tesseract language code (default: eng).",
    )
    args = parser.parse_args()

    result = predict_document_from_image(args.image_path, lang=args.lang)

    print(f"Image path      : {result['image_path']}")
    print(f"Predicted label : {result['predicted_label']}")
    print(f"OCR text length : {len(result['ocr_text'].strip())}")
    print("\nOCR text sample (first 500 chars):")
    print(result["ocr_text"][:500])


if __name__ == "__main__":
    main()