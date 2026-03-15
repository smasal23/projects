from __future__ import annotations

import numpy as np
import pandas as pd


FINAL_FEATURE_COLUMNS = [
    "elevation",
    "aspect",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
    "wilderness_area",
    "soil_type",
    "aspect_sin",
    "aspect_cos",
    "hydrology_distance",
    "hillshade_mean",
    "hillshade_std",
    "elevation_slope_interaction",
    "elevation_slope_ratio",
    "road_fire_gap",
    "road_fire_ratio",
    "hydrology_fire_ratio",
    "hydrology_road_ratio",
]


def build_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Aspect trig features
    aspect_radians = np.deg2rad(out["aspect"])
    out["aspect_sin"] = np.sin(aspect_radians)
    out["aspect_cos"] = np.cos(aspect_radians)

    # Distance-based feature
    out["hydrology_distance"] = np.sqrt(
        out["horizontal_distance_to_hydrology"] ** 2
        + out["vertical_distance_to_hydrology"] ** 2
    )

    # Hillshade aggregates
    hillshade_cols = ["hillshade_9am", "hillshade_noon", "hillshade_3pm"]
    out["hillshade_mean"] = out[hillshade_cols].mean(axis=1)
    out["hillshade_std"] = out[hillshade_cols].std(axis=1, ddof=0)

    # Interaction / ratio features
    slope_safe = out["slope"].replace(0, 1e-6)
    fire_safe = out["horizontal_distance_to_fire_points"].replace(0, 1e-6)
    road_safe = out["horizontal_distance_to_roadways"].replace(0, 1e-6)

    out["elevation_slope_interaction"] = out["elevation"] * out["slope"]
    out["elevation_slope_ratio"] = out["elevation"] / slope_safe

    out["road_fire_gap"] = (
        out["horizontal_distance_to_roadways"]
        - out["horizontal_distance_to_fire_points"]
    )
    out["road_fire_ratio"] = (
        out["horizontal_distance_to_roadways"] / fire_safe
    )
    out["hydrology_fire_ratio"] = (
        out["hydrology_distance"] / fire_safe
    )
    out["hydrology_road_ratio"] = (
        out["hydrology_distance"] / road_safe
    )

    final_df = out[FINAL_FEATURE_COLUMNS].copy()
    return final_df