from pathlib import Path

import pandas as pd

from document_checker.dataset import (
    load_dataset_index,
    subset_by_split_and_limit,
    check_dataset_paths,
)
from document_checker.ocr.tesseract import run_tesseract_on_image


def main() -> None:
    index_df = load_dataset_index()
    subset = subset_by_split_and_limit(index_df, split="train", n_per_label=2, random_state=42)

    if subset.empty:
        raise RuntimeError("Empty subset for OCR batch test")

    # Optional: check that all paths exist before running OCR
    checked = check_dataset_paths(subset)
    if not checked["exists"].all():
        missing = checked.loc[~checked["exists"], "path"].tolist()
        raise RuntimeError(f"Found non-existing paths before OCR: {missing}")

    records = []

    for _, row in subset.iterrows():
        image_path = Path(row["path"])
        label = row["label"]
        split = row["split"]

        print(f"Running Tesseract on: {image_path}")

        text = run_tesseract_on_image(image_path)

        records.append(
            {
                "path": str(image_path),
                "label": label,
                "split": split,
                "engine": "pytesseract_default",
                "text": text,
                "text_length": len(text),
            }
        )

    output_dir = Path("artifacts/ocr_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tesseract_batch_sample.csv"

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"\nSaved batch OCR results to: {output_path}")
    print("Per-label counts:")
    print(df["label"].value_counts().sort_index())
    print("\nText length stats:")
    print(df["text_length"].describe())


if __name__ == "__main__":
    main()