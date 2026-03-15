"""
build_features.py

# Goal
Create derived features (only meaningful + justified) and save engineered dataset.

# Inputs
- data/processed/X_train, X_test (or cleaned dataset before split)

# Outputs
- data/processed/X_train_fe.*
- data/processed/X_test_fe.*
- reports/feature_manifest/feature_definitions.md

# Rules
- No target leakage: never use y to compute feature values.
- Apply same transformations to train and test.

# Steps
1) Load X_train and X_test.
2) For each engineered feature:
   - define formula
   - add column
   - log why it's useful
3) Validate:
   - no new nulls
   - no inf values
4) Save engineered datasets.
5) Write feature definitions report.
"""
# Imports + Paths
from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd
import numpy as np


# Project Paths
project_root = Path.cwd()

interim_dir = project_root/"data"/"interim"
processed_dir = project_root/"data"/"processed"
docs_dir = project_root/"docs"

input_file = interim_dir/"forest_cover_cleaned.csv"
output_file = processed_dir/"Forest_cover_engineered.csv"
feature_log_json = docs_dir/"feature_definitions.json"
feature_log_md = docs_dir/"feature_definitions.md"

target_column = "cover_type"
epsilon = 1e-6


# Utility Functions
def load_data(input_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found:{input_file}")
    return pd.read_csv(input_file)

def safe_divide(a: pd.Series, b: pd.Series, epsilon: float = epsilon):
    return a / (b.abs() + epsilon)


# Feature Definitions Registry
def get_feature_definitions():
    return {
        "aspect_sin": "Sine transform of aspect in radians to represent circular direction.",
        "aspect_cos": "Cosine transform of aspect in radians to represent circular direction.",
        "hydrology_distance": (
            "Euclidean distance to hydrology computed from horizontal and vertical hydrology distances."
        ),
        "vertical_distance_to_hydrology_abs": (
            "Absolute vertical distance to hydrology, ignoring above/below sign."
        ),
        "hillshade_mean": "Mean hillshade across 9am, noon, and 3pm.",
        "hillshade_range": "Difference between maximum and minimum hillshade across the day.",
        "hillshade_std": "Standard deviation of hillshade values across the day.",
        "elevation_slope_interaction": "Interaction term between elevation and slope.",
        "elevation_slope_ratio": "Elevation divided by slope magnitude to capture terrain gradient relationship.",
        "road_fire_gap": (
            "Absolute difference between horizontal distance to roadways and fire points."
        ),
        "road_fire_ratio": (
            "Horizontal distance to roadways divided by horizontal distance to fire points."
        ),
        "hydrology_fire_ratio": (
            "Horizontal distance to hydrology divided by horizontal distance to fire points."
        ),
        "hydrology_road_ratio": (
            "Horizontal distance to hydrology divided by horizontal distance to roadways."
        ),
    }


# Feature Engineering Function
def build_features(df: pd.DataFrame):
    df = df.copy()

    required_cols = [
        "elevation",
        "aspect",
        "slope",
        "horizontal_distance_to_hydrology",
        "vertical_distance_to_hydrology",
        "horizontal_distance_to_roadways",
        "hillshade_9am",
        "hillshade_noon",
        "hillshade_3pm",
        "horizontal_distance_to_fire_points"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for feature engineering: {missing_cols}"
        )

    # Circular Transformation for Aspect
    aspect_rad = np.deg2rad(df["aspect"])
    df["aspect_sin"] = np.sin(aspect_rad)
    df["aspect_cos"] = np.cos(aspect_rad)

    # Hydrology Derived Features
    df["hydrology_distance"] = np.sqrt(df["horizontal_distance_to_hydrology"] ** 2 + df["vertical_distance_to_hydrology"] ** 2)

    df["vertical_distance_to_hydrology_abs"] = df["vertical_distance_to_hydrology"].abs()

    # Hillshade Summary Features
    hillshade_cols = ["hillshade_9am", "hillshade_noon", "hillshade_3pm"]

    df["hillshade_mean"] = df[hillshade_cols].mean(axis = 1)
    df["hillshade_range"] = df[hillshade_cols].max(axis = 1) - df[hillshade_cols].min(axis = 1)
    df["hillshade_std"] = df[hillshade_cols].std(axis = 1)

    # Terrain Interaction Figures
    df["elevation_slope_interaction"] = df["elevation"] * df["slope"]
    df["elevation_slope_ratio"] = safe_divide(df["elevation"], df["slope"])

    # Relative Accessibility
    df["road_fire_gap"] = (df["horizontal_distance_to_roadways"] - df["horizontal_distance_to_fire_points"]).abs()
    df["road_fire_ratio"] = safe_divide(df["horizontal_distance_to_roadways"], df["horizontal_distance_to_fire_points"])
    df["hydrology_fire_ratio"] = safe_divide(df["horizontal_distance_to_hydrology"], df["horizontal_distance_to_fire_points"])
    df["hydrology_road_ratio"] = safe_divide(df["horizontal_distance_to_hydrology"], df["horizontal_distance_to_roadways"])

    return df


# Feature Logging
def save_feature_logs(
        feature_defs: dict[str, str],
        json_path: Path,
        markdown_path: Path):
    with open(json_path, "w", encoding = "utf-8") as f:
        json.dump(feature_defs, f, indent = 2, ensure_ascii = False)

    lines = [
        "# Feature Definitions",
        "",
        "This file documents engineered features created in `src/features/build_features.py`.",
        "",
        "| Feature | Definition |",
        "|---|---|",
    ]

    for feature_name, definition in feature_defs.items():
        lines.append(f"| {feature_name} | {definition} |")

    markdown_path.write_text("\n".join(lines), encoding = "utf-8")


# Save Engineered Dataset
def save_engineered_data(df: pd.DataFrame, output_file: Path):
    df.to_csv(output_file, index = False)


# Main Workflow
def main():
    df = load_data(input_file)

    print("=" * 60)
    print("Loaded cleaned dataset")
    print(f"Input shape: {df.shape}")
    print("=" * 60)

    df_engineered = build_features(df)
    print("Engineered features added successfully.")
    print(f"Output shape: {df_engineered.shape}")

    new_features = [col for col in df_engineered.columns if col not in df.columns]
    print(f"Number of engineered features: {len(new_features)}")
    print("Engineered feature list:")
    for col in new_features:
        print(f"- {col}")

    save_engineered_data(df_engineered, output_file)
    save_feature_logs(
        feature_defs = get_feature_definitions(),
        json_path = feature_log_json,
        markdown_path = feature_log_md
    )

    print("=" * 60)
    print(f"Engineered dataset saved to: {output_file}")
    print(f"Feature definitions saved to: {feature_log_json}")
    print(f"Feature definitions saved to: {feature_log_md}")
    print("=" * 60)


if __name__ == "__main__":
    main()