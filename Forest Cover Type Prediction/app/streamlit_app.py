"""
streamlit_app.py

# Goal
UI for prediction demo.

# Steps
1) Set page config.
2) Load model (cache it).
3) Build input widgets for each feature.
4) On Predict:
   - collect input
   - call predict()
   - display result + confidence
5) Add sidebar:
   - model name
   - metric snapshot
   - usage instructions
6) Add error handling and validation messages.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from app.ui_components import (
    render_feature_preview,
    render_payload_preview,
    render_prediction_result,
    render_sidebar_model_info,
    show_model_load_error,
    show_prediction_error,
    show_prediction_success,
    show_validation_error,
)
from src.inference.feature import build_inference_features
from src.inference.predict import load_artifact
from src.inference.schema import SchemaValidationError, validate_raw_input


st.set_page_config(
    page_title="EcoType — Forest Cover Prediction",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_PATH = Path("models/random_forest_final_model.joblib")

MODEL_DISPLAY_NAME = "Random Forest Final Model"
PRIMARY_METRIC_NAME = "Macro F1"
PRIMARY_METRIC_VALUE = "Add final score"
SECONDARY_METRIC_NAME = "Accuracy"
SECONDARY_METRIC_VALUE = "Add final score"

CLASS_DESCRIPTIONS = {
    "Spruce/Fir": "High-elevation conifer forests, commonly found in cooler mountainous terrain.",
    "Lodgepole Pine": "Often found in mid-to-high elevation areas and common in this dataset.",
    "Ponderosa Pine": "Typically occurs in lower, drier montane zones.",
    "Cottonwood/Willow": "Usually associated with riparian or moist lowland environments.",
    "Aspen": "Broadleaf forest cover type, often in cool moist mountain areas.",
    "Douglas-fir": "Common in mountainous zones with moderate moisture and elevation.",
    "Krummholz": "Stunted subalpine vegetation near treeline and harsh alpine conditions.",
}


def safe_predict_proba(model: Any, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


@st.cache_resource
def load_model_cached(model_path: str) -> Any:
    return load_artifact(model_path)


def build_input_payload(
    elevation: int,
    aspect: int,
    slope: int,
    horizontal_distance_to_hydrology: int,
    vertical_distance_to_hydrology: int,
    horizontal_distance_to_roadways: int,
    hillshade_9am: int,
    hillshade_noon: int,
    hillshade_3pm: int,
    horizontal_distance_to_fire_points: int,
    wilderness_area: int,
    soil_type: int,
) -> dict[str, Any]:
    return {
        "elevation": elevation,
        "aspect": aspect,
        "slope": slope,
        "horizontal_distance_to_hydrology": horizontal_distance_to_hydrology,
        "vertical_distance_to_hydrology": vertical_distance_to_hydrology,
        "horizontal_distance_to_roadways": horizontal_distance_to_roadways,
        "hillshade_9am": hillshade_9am,
        "hillshade_noon": hillshade_noon,
        "hillshade_3pm": hillshade_3pm,
        "horizontal_distance_to_fire_points": horizontal_distance_to_fire_points,
        "wilderness_area": wilderness_area,
        "soil_type": soil_type,
    }


def get_prediction_details(model: Any, raw_payload: dict[str, Any]) -> dict[str, Any]:
    raw_df = validate_raw_input(raw_payload)
    feature_df = build_inference_features(raw_df)

    prediction = model.predict(feature_df)[0]

    result: dict[str, Any] = {
        "predicted_class": str(prediction),
        "feature_df": feature_df,
    }

    probabilities = safe_predict_proba(model, feature_df)
    if probabilities is not None:
        proba_row = probabilities[0]
        classes = list(model.classes_) if hasattr(model, "classes_") else []

        top_idx = int(proba_row.argmax())
        result["confidence"] = float(proba_row[top_idx])

        if classes:
            ranked = sorted(
                zip(classes, proba_row),
                key=lambda x: x[1],
                reverse=True,
            )
            result["top_classes"] = [
                {
                    "class": str(label),
                    "probability": float(prob),
                }
                for label, prob in ranked[:3]
            ]

    return result


def main() -> None:
    st.title("🌲 EcoType — Forest Cover Type Prediction")
    st.caption(
        "Predict forest cover type from cartographic and terrain-based features using the saved machine learning pipeline."
    )

    with st.sidebar:
        render_sidebar_model_info(
            model_path=str(MODEL_PATH),
            model_display_name=MODEL_DISPLAY_NAME,
            primary_metric_name=PRIMARY_METRIC_NAME,
            primary_metric_value=PRIMARY_METRIC_VALUE,
            secondary_metric_name=SECONDARY_METRIC_NAME,
            secondary_metric_value=SECONDARY_METRIC_VALUE,
        )

    try:
        model = load_model_cached(str(MODEL_PATH))
    except Exception as exc:
        show_model_load_error(str(exc))
        st.stop()

    st.subheader("Enter Feature Values")

    col1, col2, col3 = st.columns(3)

    with col1:
        elevation = st.number_input("Elevation", min_value=0, max_value=5000, value=2596, step=1)
        aspect = st.number_input("Aspect", min_value=0, max_value=360, value=51, step=1)
        slope = st.number_input("Slope", min_value=0, max_value=90, value=3, step=1)
        horizontal_distance_to_hydrology = st.number_input(
            "Horizontal Distance To Hydrology",
            min_value=0,
            value=258,
            step=1,
        )

    with col2:
        vertical_distance_to_hydrology = st.number_input(
            "Vertical Distance To Hydrology",
            min_value=-1000,
            max_value=1000,
            value=0,
            step=1,
        )
        horizontal_distance_to_roadways = st.number_input(
            "Horizontal Distance To Roadways",
            min_value=0,
            value=510,
            step=1,
        )
        hillshade_9am = st.number_input("Hillshade 9am", min_value=0, max_value=255, value=221, step=1)
        hillshade_noon = st.number_input("Hillshade Noon", min_value=0, max_value=255, value=232, step=1)

    with col3:
        hillshade_3pm = st.number_input("Hillshade 3pm", min_value=0, max_value=255, value=148, step=1)
        horizontal_distance_to_fire_points = st.number_input(
            "Horizontal Distance To Fire Points",
            min_value=0,
            value=6279,
            step=1,
        )
        wilderness_area = st.selectbox(
            "Wilderness Area (Encoded)",
            options=[1, 2, 3, 4],
            index=0,
        )
        soil_type = st.selectbox(
            "Soil Type (Encoded)",
            options=list(range(1, 41)),
            index=37,
        )

    raw_payload = build_input_payload(
        elevation=elevation,
        aspect=aspect,
        slope=slope,
        horizontal_distance_to_hydrology=horizontal_distance_to_hydrology,
        vertical_distance_to_hydrology=vertical_distance_to_hydrology,
        horizontal_distance_to_roadways=horizontal_distance_to_roadways,
        hillshade_9am=hillshade_9am,
        hillshade_noon=hillshade_noon,
        hillshade_3pm=hillshade_3pm,
        horizontal_distance_to_fire_points=horizontal_distance_to_fire_points,
        wilderness_area=wilderness_area,
        soil_type=soil_type,
    )

    predict_clicked = st.button("Predict Forest Cover Type", use_container_width=True)

    if predict_clicked:
        try:
            details = get_prediction_details(model=model, raw_payload=raw_payload)

            predicted_class = details["predicted_class"]
            confidence = details.get("confidence")
            top_classes = details.get("top_classes", [])
            feature_df = details["feature_df"]

            st.success("Prediction completed successfully.")

            show_prediction_success()

            render_prediction_result(
                predicted_class=predicted_class,
                confidence=confidence,
                top_classes=top_classes,
            )

            render_feature_preview(feature_df)
            render_payload_preview(raw_payload)


        except SchemaValidationError as exc:
            show_validation_error(str(exc))
        except Exception as exc:
            show_prediction_error(str(exc))


if __name__ == "__main__":
    main()
