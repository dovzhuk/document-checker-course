#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


FILE_ID = "1OxRYccKmPpnJEAIAMEQ_R4AnvDrjHUf3"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_PATH = DATA_DIR / "ocr_rvl_cdip.zip"
TARGET_DIR = DATA_DIR / "ocr_rvl_cdip"
TARGET_CSV = TARGET_DIR / "ocr_dataset.csv"
TARGET_IMAGES = TARGET_DIR / "images_tif"


def dataset_exists() -> bool:
    return TARGET_CSV.exists() and TARGET_IMAGES.exists()


def validate_dataset() -> None:
    if not TARGET_CSV.exists():
        raise FileNotFoundError(f"Missing file: {TARGET_CSV}")
    if not TARGET_IMAGES.exists():
        raise FileNotFoundError(f"Missing directory: {TARGET_IMAGES}")


def ensure_gdown() -> None:
    if shutil.which("gdown") is not None:
        return

    print("gdown is not installed. Installing with pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "gdown"],
        check=True,
    )


def download_archive() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_gdown()

    print(f"Downloading archive from Google Drive to: {ARCHIVE_PATH}")
    subprocess.run(
        [
            "gdown",
            FILE_ID,
            "-O",
            str(ARCHIVE_PATH),
        ],
        check=True,
    )


def unpack_archive() -> None:
    print(f"Extracting archive: {ARCHIVE_PATH}")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        zf.extractall(DATA_DIR)


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")

    if dataset_exists():
        print("Dataset already exists. Nothing to do.")
        print(f"CSV: {TARGET_CSV}")
        print(f"Images: {TARGET_IMAGES}")
        sys.exit(0)

    if TARGET_DIR.exists() and not dataset_exists():
        print(f"Directory exists but dataset is incomplete: {TARGET_DIR}")
        print("Remove it manually if it is broken, then rerun the script.")
        sys.exit(1)

    if ARCHIVE_PATH.exists():
        print(f"Archive already exists: {ARCHIVE_PATH}")
    else:
        download_archive()

    unpack_archive()
    validate_dataset()

    print("Done.")
    print(f"Dataset CSV: {TARGET_CSV}")
    print(f"Dataset images: {TARGET_IMAGES}")


if __name__ == "__main__":
    main()