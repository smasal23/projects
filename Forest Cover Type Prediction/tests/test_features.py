"""
test_features.py

# Goal
Feature engineering consistency tests.

# Steps
1) Apply build_features on train/test.
2) Assert same new columns appear.
3) Assert no NaNs introduced.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.make_dataset import standardize_columns
from src.preprocessing.clean import fix_dtypes

from src.features.build_features import build_features


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Elevation": [2500, 2600, 2700],
            "Aspect": [10, 20, 30],
            "Slope": [5, 10, 15],
            "Horizontal Distance To Hydrology": [100, 120, 140],
            "Vertical Distance To Hydrology": [-5, 0, 5],
            "Horizontal Distance To Roadways": [300, 320, 340],
            "Hillshade 9am": [200, 210, 220],
            "Hillshade Noon": [220, 225, 230],
            "Hillshade 3pm": [150, 155, 160],
            "Horizontal Distance To Fire Points": [500, 550, 600],
            "Cover Type": ["spruce", "pine", "fir"],
            "Wilderness Area": [1, 2, 1],
            "Soil Type": [10, 20, 30],
        }
    )


def test_engineered_features_present() -> None:
    df = standardize_columns(make_sample_dataframe())
    df, _ = fix_dtypes(df)

    featured = build_features(df)

    expected_cols = [
        "hydrology_distance",
        "hillshade_mean",
        "road_fire_ratio",
        "hydrology_fire_ratio"
    ]

    missing = [col for col in expected_cols if col not in featured.columns]
    assert not missing, f"Missing engineered columns: {missing}"


def test_no_nans_after_transformations() -> None:
    df = standardize_columns(make_sample_dataframe())
    df, _ = fix_dtypes(df)

    featured = build_features(df)

    assert not featured.isna().any().any(), "NaNs found after feature engineering."


def test_numeric_outputs_are_finite() -> None:
    df = standardize_columns(make_sample_dataframe())
    df, _ = fix_dtypes(df)

    featured = build_features(df)
    numeric_df = featured.select_dtypes(include=[np.number])

    assert np.isfinite(numeric_df.to_numpy()).all()