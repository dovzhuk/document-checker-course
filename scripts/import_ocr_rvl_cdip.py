#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


# Внешний корпус коллеги
EXTERNAL_DATASET_ROOT = Path.home() / "projects" / "ocr_dataset"
EXTERNAL_CSV = EXTERNAL_DATASET_ROOT / "ocr_dataset_clean_png.csv"

# Внутри проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_IMAGES_ROOT = PROJECT_ROOT / "data" / "ocr_rvl_cdip" / "images_tif"
TARGET_CSV = PROJECT_ROOT / "data" / "ocr_rvl_cdip" / "ocr_dataset.csv"


def convert_png_to_tif(src_png: Path, dst_tif: Path) -> None:
    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as img:
        # Сохраняем в TIFF без экзотических опций
        img.save(dst_tif, format="TIFF")


def main() -> None:
    if not EXTERNAL_CSV.exists():
        raise FileNotFoundError(f"External CSV not found: {EXTERNAL_CSV}")
    if not (EXTERNAL_DATASET_ROOT / "images_png").exists():
        raise FileNotFoundError(f"External images_png not found: {EXTERNAL_DATASET_ROOT / 'images_png'}")

    TARGET_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    missing = 0

    # Читаем внешний CSV и одновременно пишем итоговый CSV с путями в проекте
    with EXTERNAL_CSV.open("r", encoding="utf-8", newline="") as f_in, \
            TARGET_CSV.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["path", "label", "reference", "text"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            rel_png = Path(row["path"])        # например images_png/form/0000.png
            label = row["label"]
            reference = row["reference"]
            text = row["text"]

            src_png = EXTERNAL_DATASET_ROOT / rel_png
            # Внутри проекта кладём в images_tif с той же иерархией, но расширением .tif
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

            # В CSV путь относительно корня проекта
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