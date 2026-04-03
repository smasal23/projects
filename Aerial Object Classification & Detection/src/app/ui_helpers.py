from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from src.app.image_utils import save_pil_image


def render_sidebar_project_info(streamlit_config: Dict) -> None:
    """
    Render project summary inside sidebar.
    """
    content_cfg = streamlit_config.get("content", {})

    st.sidebar.markdown("## Project")
    st.sidebar.markdown(f"**{content_cfg.get('project_name', 'Project')}**")
    st.sidebar.caption(content_cfg.get("subtitle", ""))

    description = content_cfg.get("description", "")
    if description:
        st.sidebar.write(description)

    points = content_cfg.get("sidebar_points", [])
    if points:
        st.sidebar.markdown("### Highlights")
        for point in points:
            st.sidebar.markdown(f"- {point}")


def render_prediction_card(
    predicted_label: str,
    confidence: float,
    probabilities: Dict[str, float],
    confidence_decimals: int = 4,
) -> None:
    """
    Render classification result.
    """
    st.markdown("### Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Label", predicted_label)
    with col2:
        st.metric("Confidence", f"{confidence:.{confidence_decimals}f}")

    st.markdown("### Class Probabilities")
    prob_df = pd.DataFrame(
        [
            {"class_name": class_name, "probability": probability}
            for class_name, probability in probabilities.items()
        ]
    )
    st.dataframe(prob_df, width="stretch", hide_index=True)


def render_detection_summary(detection_payload: Dict, confidence_decimals: int = 4) -> None:
    """
    Render detection result summary.
    """
    st.markdown("### Detection Result")
    st.metric("Objects Detected", int(detection_payload["num_detections"]))

    detections = detection_payload.get("detections", [])
    if not detections:
        st.info("No objects were detected above the selected confidence threshold.")
        return

    table_df = pd.DataFrame(
        [
            {
                "label": row["label"],
                "confidence": round(float(row["confidence"]), confidence_decimals),
                "x1": round(row["xyxy"][0], 2),
                "y1": round(row["xyxy"][1], 2),
                "x2": round(row["xyxy"][2], 2),
                "y2": round(row["xyxy"][3], 2),
            }
            for row in detections
        ]
    )
    st.dataframe(table_df, width="stretch", hide_index=True)


def render_user_error(message: str) -> None:
    """
    Render user-friendly error.
    """
    st.error(message)


def render_section_divider() -> None:
    """
    Render section separator.
    """
    st.markdown("---")


def save_example_image_if_requested(image, output_path: Path, enabled: bool = True) -> Path | None:
    """
    Save image artifact for docs/reports.
    """
    if not enabled or image is None:
        return None

    save_pil_image(image=image, output_path=output_path)
    return output_path