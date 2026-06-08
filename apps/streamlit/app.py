from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_checker.pipeline.document_pipeline import predict_document_from_image


st.set_page_config(
    page_title="Document Checker Demo",
    page_icon="📄",
    layout="wide",
)

st.title("Document Checker Demo")
st.write("CPU-only demo: image -> OCR -> text -> ML classification")

demo_images: dict[str, str] = {
    "presentation / 0000001531": "data/ocr_rvl_cdip/images_tif/presentation/0000001531.tif",
    "invoice / 0000037010": "data/ocr_rvl_cdip/images_tif/invoice/0000037010.tif",
    "email / 0001451592": "data/ocr_rvl_cdip/images_tif/email/0001451592.tif",
}

source_mode = st.radio(
    "Choose input source",
    ["Demo image", "Upload image"],
    horizontal=True,
)

image_path: Path | None = None
image_caption: str = ""

if source_mode == "Demo image":
    selected_key = st.selectbox("Choose a demo image", list(demo_images.keys()))
    image_path = PROJECT_ROOT / demo_images[selected_key]
    image_caption = str(image_path.relative_to(PROJECT_ROOT))

    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        st.stop()

else:
    uploaded_file = st.file_uploader(
        "Upload document image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            image_path = Path(tmp.name)

        image_caption = f"uploaded: {uploaded_file.name}"
    else:
        st.info("Upload an image file to run OCR and classification.")
        st.stop()

with st.spinner("Running OCR and classification..."):
    result = predict_document_from_image(image_path)

ocr_text = result["ocr_text"].strip()
predicted_label = result["predicted_label"]

left_col, right_col = st.columns([1.35, 0.85], gap="large")

with left_col:
    st.subheader("Document image")
    st.image(str(image_path), caption=image_caption, width="stretch")

with right_col:
    st.subheader("Prediction")
    st.success(f"Predicted label: {predicted_label}")
    st.write(f"OCR text length: {len(ocr_text)}")

    st.divider()

    st.subheader("OCR text")
    st.text_area(
        "Recognized text",
        value=ocr_text,
        height=260,
        label_visibility="collapsed",
    )