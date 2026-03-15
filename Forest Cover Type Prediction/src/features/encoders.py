"""
encoders.py

# Goal
Define preprocessing for ML:
- numeric: impute + scale (if needed)
- categorical: impute + one-hot encode

# Inputs
- column lists (numeric_cols, categorical_cols)

# Outputs
- ColumnTransformer (preprocessor object)

# Steps
1) Detect numeric & categorical columns (or accept lists).
2) Numeric pipeline:
   - SimpleImputer(strategy="median")
   - StandardScaler()
3) Categorical pipeline:
   - SimpleImputer(strategy="most_frequent")
   - OneHotEncoder(handle_unknown="ignore")
4) Combine into ColumnTransformer.
5) Provide helper functions:
   - get_preprocessor(X_train)
   - get_feature_names(preprocessor) (optional)
6) Add a simple test function to verify transform works.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

target_column = 'cover_type'
categorical_candidates = ['wilderness_area', 'soil_type']


# Identify Numeric vs Categorical Column
def identify_feature_columns(
        df: pd.DataFrame,
        target_col: str = target_column,
        categorical_candidates: list[str] | None = None):

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    categorical_candidates = categorical_candidates

    feature_cols = [col for col in df.columns if col != target_col]

    categorical_cols: list[str] = []
    for col in categorical_candidates:
        if col in feature_cols:
            categorical_cols.append(col)

    inferred_cats = [
        col
        for col in feature_cols
        if str(df[col].dtype) in {"object", "category", "bool"}
    ]

    categorical_cols = sorted(set(categorical_cols + inferred_cats))
    numeric_cols = sorted([col for col in feature_cols if col not in categorical_cols])

    return numeric_cols, categorical_cols, feature_cols


# Build ColumnTransformer + encoders
def build_preprocessor(
        numeric_cols: list[str],
        categorical_cols: list[str]):

    numeric_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown = "ignore",
                    sparse_output = False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder = "drop",
        sparse_threshold = 0.0,
        verbose_feature_names_out = False
    )

    return preprocessor


# Fit on train only, transform train/test
def fit_transform_train_test(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = target_column,
        categorical_candidates: list[str] | None = None):

    for name, frame in {"train_df": train_df, "test_df": test_df}.items():
        if target_col not in frame.columns:
            raise ValueError(f"'{target_col}' missing in {name}.")

    numeric_cols, categorical_cols, feature_cols = identify_feature_columns(
        df = train_df,
        target_col = target_col,
        categorical_candidates = categorical_candidates
    )

    preprocessor = build_preprocessor(
        numeric_cols = numeric_cols,
        categorical_cols = categorical_cols
    )

    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()
    Y_train = train_df[target_col].copy()
    Y_test = test_df[target_col].copy()

    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    output_cols = preprocessor.get_feature_names_out().tolist()

    X_train = pd.Dataframe(
        X_train_arr,
        columns = output_cols,
        index = test_df.index
    )

    X_test = pd.DataFrame(
        X_test_arr,
        columns = output_cols,
        index = test_df.index
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_cols": feature_cols,
        "output_cols": output_cols
    }


# Save fitted preprocessor
def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(preprocessor, path)
    return path


# Load fitted preprocessor
def load_preprocessor(path: str | Path):
    return joblib.load(path)

# Save columns manifest
def save_columns_manifest(
        path: str | Path,
        *,
        target_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
        feature_cols: list[str],
        output_cols: list[str]
    ):

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    manifest = {
        "target_column": target_col,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "feature_columns_raw": feature_cols,
        "feature_columns_transformed": output_cols,
        "n_raw_features": len(feature_cols),
        "n_transformed_features": len(output_cols)
    }

    path.write_text(json.dumps(manifest, indent = 2), encoding = 'utf-8')
    return path