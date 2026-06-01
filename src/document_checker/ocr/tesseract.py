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