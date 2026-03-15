"""
ui_components.py

# Goal
Reusable UI components for the Streamlit app.

# Steps
1) Build input widget helpers:
   - numeric_input(field_name, default, min/max)
   - categorical_select(field_name, options)
2) Build output display helpers:
   - prediction_card(label, confidence)
   - error_banner(message)
3) Keep formatting consistent across app pages.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


CLASS_DESCRIPTIONS = {
    "Spruce/Fir": "High-elevation conifer forests, commonly found in cooler mountainous terrain.",
    "Lodgepole Pine": "Often found in mid-to-high elevation areas and common in this dataset.",
    "Ponderosa Pine": "Typically occurs in lower, drier montane zones.",
    "Cottonwood/Willow": "Usually associated with riparian or moist lowland environments.",
    "Aspen": "Broadleaf forest cover type, often in cool moist mountain areas.",
    "Douglas-fir": "Common in mountainous zones with moderate moisture and elevation.",
    "Krummholz": "Stunted subalpine vegetation near treeline and harsh alpine conditions.",
}


def render_sidebar_model_info(
    model_path: str,
    model_display_name: str,
    primary_metric_name: str,
    primary_metric_value: str,
    secondary_metric_name: str,
    secondary_metric_value: str,
) -> None:
    st.header("Model Overview")
    st.write(f"**Model:** {model_display_name}")
    st.write(f"**Artifact path:** `{model_path}`")
    st.write(f"**Primary metric:** {primary_metric_name}")
    st.write(f"**Primary score:** {primary_metric_value}")
    st.write(f"**Secondary metric:** {secondary_metric_name}")
    st.write(f"**Secondary score:** {secondary_metric_value}")
    st.divider()
    st.subheader("Input Notes")
    st.markdown(
        """
        - `wilderness_area` should be an encoded category from **1 to 4**
        - `soil_type` should be an encoded category from **1 to 40**
        - Distances are expected in the same scale used during training
        - Hillshade values should remain between **0 and 255**
        """
    )


def render_prediction_result(
    predicted_class: str,
    confidence: float | None = None,
    top_classes: list[dict[str, Any]] | None = None,
) -> None:
    result_col, info_col = st.columns([1.3, 1])

    with result_col:
        st.markdown("### Predicted Cover Type")
        st.metric(label="Predicted Class", value=predicted_class)
        if confidence is not None:
            st.metric(label="Confidence", value=f"{confidence:.2%}")

        description = CLASS_DESCRIPTIONS.get(predicted_class)
        if description:
            st.info(description)

    with info_col:
        if top_classes:
            st.markdown("### Top Class Probabilities")
            top_df = pd.DataFrame(top_classes)
            top_df["probability"] = top_df["probability"].map(lambda x: f"{x:.2%}")
            st.dataframe(top_df, use_container_width=True, hide_index=True)


def render_feature_preview(feature_df: pd.DataFrame) -> None:
    with st.expander("See transformed features used for prediction"):
        st.dataframe(feature_df, use_container_width=True, hide_index=True)


def render_payload_preview(raw_payload: dict[str, Any]) -> None:
    with st.expander("See raw input payload"):
        st.json(raw_payload)


def show_validation_error(message: str) -> None:
    st.error(f"Input validation failed: {message}")


def show_prediction_error(message: str) -> None:
    st.error(f"Prediction failed: {message}")


def show_model_load_error(message: str) -> None:
    st.error(f"Failed to load model artifact: {message}")


def show_prediction_success() -> None:
    st.success("Prediction completed successfully.")
