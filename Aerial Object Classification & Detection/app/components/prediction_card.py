from __future__ import annotations

from src.app.ui_helpers import render_detection_summary, render_prediction_card


def show_classification_prediction(prediction_payload: dict, confidence_decimals: int = 4) -> None:
    """
    Render classification card.
    """
    render_prediction_card(
        predicted_label=prediction_payload["predicted_label"],
        confidence=prediction_payload["confidence"],
        probabilities=prediction_payload["probabilities"],
        confidence_decimals=confidence_decimals,
    )


def show_detection_prediction(detection_payload: dict, confidence_decimals: int = 4) -> None:
    """
    Render detection card.
    """
    render_detection_summary(
        detection_payload=detection_payload,
        confidence_decimals=confidence_decimals,
    )