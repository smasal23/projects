from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.components.prediction_card import show_detection_prediction
from app.components.uploader import show_image_uploader
from src.app.image_utils import (
    get_image_display_metadata,
    load_pil_image_from_upload,
    resize_for_display,
    save_uploaded_image,
    validate_uploaded_image,
)
from src.app.inference_detector import load_detector, run_detector_on_image
from src.app.ui_helpers import render_section_divider, render_user_error, save_example_image_if_requested
from src.utils.config import load_project_configs


CONFIGS = load_project_configs(PROJECT_ROOT / "configs")
STREAMLIT_CFG = CONFIGS["streamlit"]

UI_CFG = STREAMLIT_CFG.get("ui", {})
DETECTOR_UI_CFG = STREAMLIT_CFG.get("detection", {})
ARTIFACT_CFG = STREAMLIT_CFG.get("artifacts", {})


@st.cache_resource(show_spinner=False)
def get_loaded_detector(model_path: str):
    return load_detector(Path(model_path))


st.title("Detection")
st.caption("Upload an image and run YOLOv8 detection.")

detector_model_path = PROJECT_ROOT / DETECTOR_UI_CFG.get("detector_model_path", "models/detection/final/best_detector.pt")

if not DETECTOR_UI_CFG.get("enabled", True):
    render_user_error("Detection mode is disabled in streamlit config.")
    st.stop()

if not detector_model_path.exists():
    render_user_error(f"Detector model not found: {detector_model_path}")
    st.stop()

uploaded_file = show_image_uploader()

if uploaded_file is None:
    st.info("Upload an image to run detection.")
    st.stop()

try:
    validate_uploaded_image(uploaded_file, max_size_mb=int(UI_CFG.get("max_upload_size_mb", 10)))
    pil_image = load_pil_image_from_upload(uploaded_file)
    display_image = resize_for_display(pil_image)
    image_meta = get_image_display_metadata(pil_image)
except Exception as exc:
    render_user_error(f"Image validation failed: {exc}")
    st.stop()

col_preview, col_meta = st.columns([2, 1])

with col_preview:
    st.markdown("### Preview")
    st.image(display_image, use_container_width=True)

with col_meta:
    st.markdown("### Image Details")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Width:** {image_meta['width']}")
    st.write(f"**Height:** {image_meta['height']}")
    st.write(f"**Aspect Ratio:** {image_meta['aspect_ratio']}")
    st.write(f"**Mode:** {image_meta['mode']}")

render_section_divider()

conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(DETECTOR_UI_CFG.get("confidence_threshold", 0.25)),
    step=0.05,
)

temp_upload_dir = PROJECT_ROOT / ARTIFACT_CFG.get("temp_upload_dir", "data/interim/app_uploads")
saved_upload_path = save_uploaded_image(
    uploaded_file=uploaded_file,
    output_dir=temp_upload_dir,
    output_name="detection_current_upload",
)

with st.spinner("Running YOLOv8 detection..."):
    try:
        detector = get_loaded_detector(str(detector_model_path))
        detection_payload = run_detector_on_image(
            detector=detector,
            image_path=saved_upload_path,
            conf=conf_threshold,
            line_width=int(DETECTOR_UI_CFG.get("line_width", 2)),
        )

        st.markdown("### Annotated Detection Output")
        st.image(detection_payload["annotated_image"], use_container_width=True)

        show_detection_prediction(
            detection_payload=detection_payload,
            confidence_decimals=int(UI_CFG.get("confidence_decimals", 4)),
        )

        if UI_CFG.get("save_example_assets", True):
            detector_output_path = PROJECT_ROOT / ARTIFACT_CFG.get(
                "detector_example_path",
                "figures/app/streamlit_detector_output.png",
            )
            docs_detection_path = PROJECT_ROOT / ARTIFACT_CFG.get(
                "docs_detection_path",
                "docs/images/streamlit_detection.png",
            )

            save_example_image_if_requested(
                image=detection_payload["annotated_image"],
                output_path=detector_output_path,
                enabled=True,
            )
            save_example_image_if_requested(
                image=detection_payload["annotated_image"],
                output_path=docs_detection_path,
                enabled=True,
            )

    except Exception as exc:
        render_user_error(f"Detection inference failed: {exc}")