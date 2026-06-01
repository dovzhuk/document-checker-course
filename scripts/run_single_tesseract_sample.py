from pathlib import Path

import pandas as pd

from document_checker.dataset import (
    load_dataset_index,
    subset_by_split_and_limit,
)
from document_checker.ocr.tesseract import run_tesseract_on_image


def main() -> None:
    index_df = load_dataset_index()
    subset = subset_by_split_and_limit(index_df, split="train", n_per_label=1, random_state=42)

    if subset.empty:
        raise RuntimeError("Empty subset for OCR test")

    row = subset.iloc[0]

    image_path = Path(row["path"])
    label = row["label"]
    split = row["split"]

    print(f"Running Tesseract on: {image_path}")

    text = run_tesseract_on_image(image_path)

    output_dir = Path("artifacts/ocr_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tesseract_single_sample.csv"

    df = pd.DataFrame(
        [
            {
                "path": str(image_path),
                "label": label,
                "split": split,
                "engine": "pytesseract_default",
                "text": text,
            }
        ]
    )

    df.to_csv(output_path, index=False)

    print(f"OCR result saved to: {output_path}")
    print(f"Extracted text length: {len(text)}")


if __name__ == "__main__":
    main()