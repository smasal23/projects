"""
schema.py

# Goal
Define expected input schema for inference.

# Outputs
- validate_input(df) that:
  - checks required columns exist
  - coerces dtypes safely
  - rejects/flags invalid ranges
  - returns cleaned single-row DataFrame

# Steps
1) Define REQUIRED_FEATURES list.
2) Define FEATURE_DTYPES mapping.
3) Define optional FEATURE_RANGES mapping.
4) Implement validate_input(input_df).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

@dataclass(frozen = True)
class FeatureSchema:
    required_columns: list[str]
    numeric_columns: list[str]
    ranges: dict[str, tuple[float | None, float | None]]

raw_numeric_columns = [
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

raw_feature_ranges = {
    "elevation": (0, 5000),
    "aspect": (0, 360),
    "slope": (0, 90),
    "horizontal_distance_to_hydrology": (0, None),
    "vertical_distance_to_hydrology": (-1000, 1000),
    "horizontal_distance_to_roadways": (0, None),
    "hillshade_9am": (0, 255),
    "hillshade_noon": (0, 255),
    "hillshade_3pm": (0, 255),
    "horizontal_distance_to_fire_points": (0, None),
    "wilderness_area": (1, 4),
    "soil_type": (1, 40)
}

schema = FeatureSchema(
    required_columns = raw_numeric_columns,
    numeric_columns = raw_numeric_columns,
    ranges = raw_feature_ranges
)

class SchemaValidationError(ValueError):
    """Raised when inference input fails schema validation."""

def _ensure_dataframe(data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise SchemaValidationError(
        "Input must be a pandas DataFrame, a dict, or a list of dicts."
    )


def validate_raw_input(data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]], schema: FeatureSchema = schema):
    df = _ensure_dataframe(data)

    missing_cols = [col for col in schema.required_columns if col not in df.columns]
    if missing_cols:
        raise SchemaValidationError(
            f"Missing required columns: {missing_cols}"
        )

    # Keep only required columns in training order
    df = df[schema.required_columns].copy()

    # Numeric conversion
    for col in schema.numeric_columns:
        df[col] = pd.to_numeric(df[col], errors = "coerce")

    # Check missing after conversion
    bad_nulls = df.isnull().sum()
    bad_nulls = bad_nulls[bad_nulls > 0]
    if not bad_nulls.empty:
        raise SchemaValidationError(
            f"Null/invalid values found after type conversion: {bad_nulls.to_dict()}"
        )

    # Range Checks
    for col,(min_val, max_val) in schema.ranges.items():
        if min_val is not None and (df[col] < min_val).any():
            bad_rows = df.index[df[col] < min_val].tolist()
            raise SchemaValidationError(
                f"Column '{col}' has values below {min_val} at rows {bad_rows}"
            )

        if max_val is not None and (df[col] > max_val).any():
            bad_rows = df.index[df[col] > max_val].tolist()
            raise SchemaValidationError(
                f"Column '{col}' has values above {max_val} at rows {bad_rows}"
            )

    return df