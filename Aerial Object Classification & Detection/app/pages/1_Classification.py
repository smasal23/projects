from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.components.prediction_card import show_classification_prediction
from app.components.uploader import show_image_uploader
from src.app.image_utils import (
    get_image_display_metadata,
    load_pil_image_from_upload,
    resize_for_display,
    save_uploaded_image,
    validate_uploaded_image,
)
from src.app.inference_classifier import (
    load_class_mapping,
    load_keras_model,
    predict_with_loaded_classifier,
    resolve_enabled_classifier_registry,
)
from src.app.ui_helpers import render_section_divider, render_user_error, save_example_image_if_requested
from src.utils.config import load_project_configs


CONFIGS = load_project_configs(PROJECT_ROOT / "configs")
STREAMLIT_CFG = CONFIGS["streamlit"]
CLASSIFICATION_CFG = CONFIGS["classification"]

UI_CFG = STREAMLIT_CFG.get("ui", {})
CLASSIFIER_UI_CFG = STREAMLIT_CFG.get("classification", {})
ARTIFACT_CFG = STREAMLIT_CFG.get("artifacts", {})


@st.cache_resource(show_spinner=False)
def get_classifier_registry():
    return resolve_enabled_classifier_registry(
        project_root=PROJECT_ROOT,
        streamlit_config=STREAMLIT_CFG,
    )


@st.cache_resource(show_spinner=False)
def get_loaded_classifier(model_path: str):
    return load_keras_model(Path(model_path))


st.title("Classification")
st.caption("Upload an image and classify it as bird or drone.")

uploaded_file = show_image_uploader()

if uploaded_file is None:
    st.info("Upload an image to run classification.")
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

try:
    registry = get_classifier_registry()
except Exception as exc:
    render_user_error(f"Classifier registry could not be built: {exc}")
    st.stop()

if not registry:
    render_user_error("No valid classifier artifacts were found.")
    st.stop()

classifier_keys = list(registry.keys())
default_model_key = CLASSIFIER_UI_CFG.get("default_model_key", classifier_keys[0])
default_index = classifier_keys.index(default_model_key) if default_model_key in classifier_keys else 0

if UI_CFG.get("enable_model_selector", True):
    selected_model_key = st.selectbox(
        "Select Classification Model",
        options=classifier_keys,
        index=default_index,
        format_func=lambda key: registry[key]["display_name"],
    )
else:
    selected_model_key = classifier_keys[default_index]

selected_entry = registry[selected_model_key]

st.write(f"**Selected model:** {selected_entry['display_name']}")
st.write(f"**Preprocessing mode:** {selected_entry['preprocess_mode']}")
if selected_entry.get("backbone_name"):
    st.write(f"**Backbone:** {selected_entry['backbone_name']}")

temp_upload_dir = PROJECT_ROOT / ARTIFACT_CFG.get("temp_upload_dir", "data/interim/app_uploads")
saved_upload_path = save_uploaded_image(
    uploaded_file=uploaded_file,
    output_dir=temp_upload_dir,
    output_name="classification_current_upload",
)

with st.spinner("Running classifier inference..."):
    try:
        classifier_model = get_loaded_classifier(str(selected_entry["model_path"]))
        class_mapping = load_class_mapping(Path(selected_entry["class_mapping_path"]))

        prediction_payload = predict_with_loaded_classifier(
            model=classifier_model,
            image=pil_image,
            class_mapping=class_mapping,
            preprocess_mode=selected_entry["preprocess_mode"],
            backbone_name=selected_entry.get("backbone_name"),
            image_size=tuple(CLASSIFICATION_CFG["custom_cnn"]["training"]["image_size"]),
        )

        show_classification_prediction(
            prediction_payload=prediction_payload,
            confidence_decimals=int(UI_CFG.get("confidence_decimals", 4)),
        )

        if UI_CFG.get("save_example_assets", True):
            classifier_output_path = PROJECT_ROOT / ARTIFACT_CFG.get(
                "classifier_example_path",
                "figures/app/streamlit_classifier_output.png",
            )
            docs_prediction_path = PROJECT_ROOT / ARTIFACT_CFG.get(
                "docs_prediction_path",
                "docs/images/streamlit_prediction.png",
            )

            save_example_image_if_requested(
                image=pil_image,
                output_path=classifier_output_path,
                enabled=True,
            )
            save_example_image_if_requested(
                image=pil_image,
                output_path=docs_prediction_path,
                enabled=True,
            )

    except Exception as exc:
        render_user_error(f"Classification inference failed: {exc}")