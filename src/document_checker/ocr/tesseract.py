from pathlib import Path

from PIL import Image
import pytesseract


def run_tesseract_on_image(image_path: Path | str) -> str:
    """Run Tesseract OCR on a single image and return extracted text.

    Parameters
    ----------
    image_path : Path | str
        Path to the image file (e.g. .tif).

    Returns
    -------
    str
        Extracted text (may be empty if OCR finds nothing).
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {path}")

    # Open image with PIL and run Tesseract
    with Image.open(path) as img:
        text = pytesseract.image_to_string(img)

    return text


def run_tesseract_on_image_psm6(
    image_path: Path | str,
    lang: str = "eng",
) -> str:
    """Run Tesseract OCR on a single image with PSM 6 configuration.

    This uses a slightly different page segmentation mode, which may work
    better for some document layouts.

    Parameters
    ----------
    image_path : Path | str
        Path to the image file (e.g. .tif).
    lang : str, optional
        Tesseract language code, by default "eng".

    Returns
    -------
    str
        Extracted text (may be empty if OCR finds nothing).
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {path}")

    with Image.open(path) as img:
        # PSM 6: Assume a single uniform block of text.
        config = "--psm 6"
        text = pytesseract.image_to_string(img, lang=lang, config=config)

    return text


def run_tesseract_on_image_sparse_lstm(
    image_path: Path | str,
    lang: str = "eng",
) -> str:
    """Run Tesseract OCR in 'sparse text' LSTM mode (PSM 11, OEM 1).

    This configuration is more aggressive in finding text regions and uses
    the LSTM engine, which can improve accuracy on some complex documents
    at the cost of speed.

    Parameters
    ----------
    image_path : Path | str
        Path to the image file (e.g. .tif).
    lang : str, optional
        Tesseract language code, by default "eng".

    Returns
    -------
    str
        Extracted text (may be empty if OCR finds nothing).
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {path}")

    with Image.open(path) as img:
        # PSM 11: Sparse text, OEM 1: LSTM only.
        config = "--psm 11 --oem 1"
        text = pytesseract.image_to_string(img, lang=lang, config=config)

    return text