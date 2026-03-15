"""
test_data_pipeline.py

# Goal
Ensure data pipeline runs without breaking.

# Steps
1) Load a small sample dataset.
2) Run clean.py functions.
3) Run split.py.
4) Assert expected outputs exist and shapes make sense.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.make_dataset import standardize_columns, validate_target
from src.preprocessing.clean import validate_columns, fix_dtypes
from src.preprocessing.split import split_features_target, perform_train_test_split


TARGET_COL = "cover_type"


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Elevation": [2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600],
            "Aspect": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            "Slope": [5, 10, 15, 20, 25, 30, 35, 12, 18, 22, 28, 32],
            "Horizontal Distance To Hydrology": [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320],
            "Vertical Distance To Hydrology": [-5, 0, 5, 10, -2, 3, 8, -1, 4, 7, 2, 6],
            "Horizontal Distance To Roadways": [300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520],
            "Hillshade 9am": [200, 210, 220, 230, 240, 250, 245, 235, 225, 215, 205, 195],
            "Hillshade Noon": [220, 225, 230, 235, 240, 245, 250, 248, 242, 238, 232, 228],
            "Hillshade 3pm": [150, 155, 160, 165, 170, 175, 180, 178, 172, 168, 162, 158],
            "Horizontal Distance To Fire Points": [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
            "Cover Type": [
                "spruce", "pine", "fir",
                "spruce", "pine", "fir",
                "spruce", "pine", "fir",
                "spruce", "pine", "fir"
            ],
            "Wilderness Area": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "Soil Type": [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30],
        }
    )


def test_data_load_clean_steps_do_not_break() -> None:
    df = make_sample_dataframe()

    df = standardize_columns(df)
    target_col = validate_target(df)

    assert target_col == TARGET_COL

    validate_columns(df)

    cleaned_df, dtype_changes = fix_dtypes(df)

    assert isinstance(cleaned_df, pd.DataFrame)
    assert not cleaned_df.empty
    assert TARGET_COL in cleaned_df.columns
    assert cleaned_df[TARGET_COL].dtype.name == "string"
    assert isinstance(dtype_changes, dict)


def test_split_reproducibility() -> None:
    df = standardize_columns(make_sample_dataframe())
    df, _ = fix_dtypes(df)

    X, y = split_features_target(df, TARGET_COL)

    X_train_1, X_test_1, y_train_1, y_test_1 = perform_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = perform_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pd.testing.assert_frame_equal(
        X_train_1.reset_index(drop=True),
        X_train_2.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        X_test_1.reset_index(drop=True),
        X_test_2.reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        y_train_1.reset_index(drop=True),
        y_train_2.reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        y_test_1.reset_index(drop=True),
        y_test_2.reset_index(drop=True),
    )