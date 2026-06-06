#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


# Внешний корпус v2 (jsonl + PNG)
EXTERNAL_DATASET_ROOT = Path.home() / "projects" / "ocr_dataset_v2" / "ocr_dataset"
EXTERNAL_JSONL = EXTERNAL_DATASET_ROOT / "ocr_dataset.jsonl"

# Внутри проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_IMAGES_ROOT = PROJECT_ROOT / "data" / "ocr_rvl_cdip" / "images_tif"
TARGET_CSV = PROJECT_ROOT / "data" / "ocr_rvl_cdip" / "ocr_dataset.csv"


def convert_png_to_tif(src_png: Path, dst_tif: Path) -> None:
    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as img:
        img.save(dst_tif, format="TIFF")


def main() -> None:
    if not EXTERNAL_JSONL.exists():
        raise FileNotFoundError(f"External jsonl not found: {EXTERNAL_JSONL}")
    if not (EXTERNAL_DATASET_ROOT / "images_png").exists():
        raise FileNotFoundError(f"External images_png not found: {EXTERNAL_DATASET_ROOT / 'images_png'}")

    TARGET_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    missing = 0

    # Пишем итоговый CSV в том же формате, что и ранее
    # path, label, reference, text
    import csv

    with TARGET_CSV.open("w", encoding="utf-8", newline="") as f_out:
        fieldnames = ["path", "label", "reference", "text"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        with EXTERNAL_JSONL.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)

                label = record["class"]
                reference = record["id"]          # например "budget/0000009955"
                rel_png = Path(record["image"])  # например images_png/budget/0000009955.png
                text = record["text"]

                src_png = EXTERNAL_DATASET_ROOT / rel_png

                # Внутри проекта кладём в images_tif с той же иерархией, но .tif
                rel_tif = Path("images_tif") / rel_png.relative_to("images_png")
                rel_tif = rel_tif.with_suffix(".tif")
                dst_tif = PROJECT_ROOT / "data" / "ocr_rvl_cdip" / rel_tif

                if not src_png.exists():
                    print(f"Missing PNG: {src_png}")
                    missing += 1
                    continue

                if not dst_tif.exists():
                    convert_png_to_tif(src_png, dst_tif)
                    converted += 1

                writer.writerow(
                    {
                        "path": str(rel_tif),
                        "label": label,
                        "reference": reference,
                        "text": text,
                    }
                )

    print(f"Converted PNG → TIFF: {converted}")
    print(f"Missing PNG files: {missing}")
    print(f"Target CSV written to: {TARGET_CSV}")
    print(f"Images stored under: {TARGET_IMAGES_ROOT}")


if __name__ == "__main__":
    main()