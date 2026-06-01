from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path("artifacts/ocr_eval/tesseract_three_configs_batch.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"OCR results file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Берём только один движок, чтобы не дублировать пути (например, default)
    base_df = df[df["engine"] == "pytesseract_default"].copy()

    # Вариант 1: по одному документу на каждый класс (если классов не слишком много)
    # Если хочешь меньше — можно потом вручную удалить лишние строки в CSV.
    base_df_sorted = (
        base_df.sort_values(["label", "path"])
        .groupby("label", as_index=False)
        .head(1)
    )

    # На случай, если хочешь ограничить общее число документов:
    # base_df_sorted = base_df_sorted.head(20)

    output_dir = Path("artifacts/ocr_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manual_annotation_list.csv"

    base_df_sorted[["path", "label"]].to_csv(output_path, index=False)

    print(f"Saved manual annotation list to: {output_path}")
    print("Number of documents to annotate:", len(base_df_sorted))
    print(base_df_sorted.head())


if __name__ == "__main__":
    main()