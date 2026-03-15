"""
test_inference.py

# Goal
Inference validation tests.

# Steps
1) Create sample valid input.
2) validate_input should pass.
3) predict() should return a valid class.
4) Invalid input should raise helpful error.
"""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_model.pkl"


REQUIRED_COLUMNS = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
    "wilderness_area",
    "soil_type",
]


def validate_input_schema(payload: dict) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in payload]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return pd.DataFrame([payload], columns=REQUIRED_COLUMNS)


def make_valid_payload() -> dict:
    return {
        "elevation": 2596,
        "aspect": 51,
        "slope": 3,
        "horizontal_distance_to_hydrology": 258,
        "vertical_distance_to_hydrology": 0,
        "horizontal_distance_to_roadways": 510,
        "hillshade_9am": 221,
        "hillshade_noon": 232,
        "hillshade_3pm": 148,
        "horizontal_distance_to_fire_points": 6279,
        "wilderness_area": 1,
        "soil_type": 10,
    }


def test_schema_validation_works_for_valid_payload() -> None:
    payload = make_valid_payload()
    df = validate_input_schema(payload)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, len(REQUIRED_COLUMNS))
    assert list(df.columns) == REQUIRED_COLUMNS


def test_schema_validation_fails_for_missing_column() -> None:
    payload = make_valid_payload()
    payload.pop("elevation")

    with pytest.raises(ValueError):
        validate_input_schema(payload)


def test_predict_returns_expected_types() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    payload = make_valid_payload()
    df = validate_input_schema(payload)

    prediction = model.predict(df)

    assert len(prediction) == 1
    assert isinstance(prediction[0], (str, int, float))