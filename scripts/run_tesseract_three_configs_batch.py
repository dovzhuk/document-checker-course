from pathlib import Path

import pandas as pd

from document_checker.dataset import (
    load_dataset_index,
    subset_by_split_and_limit,
    check_dataset_paths,
)
from document_checker.ocr.tesseract import (
    run_tesseract_on_image,
    run_tesseract_on_image_psm6,
    run_tesseract_on_image_sparse_lstm,
)


def main() -> None:
    index_df = load_dataset_index()
    subset = subset_by_split_and_limit(
        index_df,
        split="train",
        n_per_label=2,
        random_state=42,
    )

    if subset.empty:
        raise RuntimeError("Empty subset for OCR batch test")

    checked = check_dataset_paths(subset)
    if not checked["exists"].all():
        missing = checked.loc[~checked["exists"], "path"].tolist()
        raise RuntimeError(f"Found non-existing paths before OCR: {missing}")

    engines = [
        ("pytesseract_default", run_tesseract_on_image),
        ("pytesseract_psm6", run_tesseract_on_image_psm6),
        ("pytesseract_sparse_lstm", run_tesseract_on_image_sparse_lstm),
    ]

    records = []

    for _, row in subset.iterrows():
        image_path = Path(row["path"])
        label = row["label"]
        split = row["split"]

        for engine_name, ocr_func in engines:
            print(f"Running {engine_name} on: {image_path}")
            text = ocr_func(image_path)

            records.append(
                {
                    "path": str(image_path),
                    "label": label,
                    "split": split,
                    "engine": engine_name,
                    "text": text,
                    "text_length": len(text),
                }
            )

    output_dir = Path("artifacts/ocr_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tesseract_three_configs_batch.csv"

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"\nSaved OCR results to: {output_path}")
    print("\nCounts by engine:")
    print(df["engine"].value_counts())
    print("\nCounts by label and engine:")
    print(df.groupby(["label", "engine"]).size())
    print("\nText length stats by engine:")
    print(df.groupby("engine")["text_length"].describe())


if __name__ == "__main__":
    main()