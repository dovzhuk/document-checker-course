from pathlib import Path

import pandas as pd

from document_checker.ocr.metrics import (
    character_accuracy,
    word_accuracy,
    word_error_rate,
)


def main() -> None:
    ocr_results_path = Path("artifacts/ocr_eval/tesseract_three_configs_batch.csv")
    gt_path = Path("artifacts/ocr_eval/manual_ground_truth.csv")

    if not ocr_results_path.exists():
        raise FileNotFoundError(f"OCR results file not found: {ocr_results_path}")

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    ocr_df = pd.read_csv(ocr_results_path)
    gt_df = pd.read_csv(gt_path)

    # Оставляем только те строки OCR, для которых есть эталон
    merged = ocr_df.merge(
        gt_df[["path", "label", "reference_text"]],
        on=["path", "label"],
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        raise RuntimeError("No overlapping records between OCR results and ground truth.")

    records: list[dict] = []

    for _, row in merged.iterrows():
        path = row["path"]
        label = row["label"]
        engine = row["engine"]
        ocr_text = row["text"] if isinstance(row["text"], str) else ""
        ref_text = row["reference_text"] if isinstance(row["reference_text"], str) else ""

        char_acc = character_accuracy(ref_text, ocr_text)
        word_acc = word_accuracy(ref_text, ocr_text)
        wer = word_error_rate(ref_text, ocr_text)

        records.append(
            {
                "path": path,
                "label": label,
                "engine": engine,
                "char_accuracy": char_acc,
                "word_accuracy": word_acc,
                "wer": wer,
            }
        )

    results_df = pd.DataFrame(records)

    output_dir = Path("artifacts/ocr_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ocr_evaluation_results.csv"

    results_df.to_csv(output_path, index=False)

    print(f"Saved OCR evaluation results to: {output_path}\n")

    print("Per-engine mean metrics:")
    print(
        results_df.groupby("engine")[["char_accuracy", "word_accuracy", "wer"]].mean()
    )

    print("\nPer-engine median metrics:")
    print(
        results_df.groupby("engine")[["char_accuracy", "word_accuracy", "wer"]].median()
    )

    print("\nNumber of documents per engine:")
    print(results_df["engine"].value_counts())


if __name__ == "__main__":
    main()