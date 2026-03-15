"""
selection.py

# Goal
Remove low-quality features and freeze final feature list.

# Inputs
- X_train_fe, X_test_fe

# Outputs
- X_train_final, X_test_final
- reports/feature_manifest/final_features.txt

# Steps
1) Drop constant features.
2) Drop near-zero variance features (optional threshold).
3) Drop duplicate columns (if any).
4) (Optional) correlation pruning with justification.
5) Save final feature list to text file.
6) Save transformed train/test with only final features.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from patsy.user_util import balanced

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from src.features.build_features import output_file


# Project Paths + Config
project_root = Path.cwd()

processed_dir = project_root/"data"/"processed"
docs_dir = project_root/"docs"

input_file = processed_dir/"forest_cover_engineered.csv"
output_file = processed_dir/"forest_cover_selected.csv"
final_features_file = docs_dir/"final_features.txt"
selection_summary_file = docs_dir/"feature_selection_summary.md"

target_column = "cover_type"

near_constant_threshold = 0.995
correlation_threshold = 0.95
importance_threshold = 0.001

random_state = 42
test_size = 0.2


# Utility Function
def load_data(input_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    return pd.read_csv(input_file)


# Split Features and Target
def split_features_target(df: pd.DataFrame,target_col: str = target_column):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns = [target_col]).copy()
    Y = df[target_col].copy()
    return X, Y


# Constant / Near-Constant Filtering
def get_constant_features(X: pd.DataFrame):
    constant_features = [col for col in X.columns if X[col].nunique(dropna = False) <= 1]
    return constant_features

def get_near_constant_features(X: pd.DataFrame, threshold: float = near_constant_threshold):
    near_constant_features = []

    for col in X.columns:
        top_freq = X[col].value_counts(dropna = False, normalize = True).iloc[0]
        if top_freq >= threshold:
            near_constant_features.append(col)

    return near_constant_features


# Correlation Pruning
def get_correlated_feature_drops(X: pd.DataFrame, Y: pd.Series, threshold: float = correlation_threshold):
    numeric_cols = X.select_dtypes(include = [np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return[]

    X_num = X[numeric_cols].copy()
    corr_matrix = X_num.corr().abs()

    mi_scores = mutual_info_classif(
        X_num.fillna(X_num.median()),
        Y,
        random_state = random_state
    )
    mi_dict = dict(zip(numeric_cols, mi_scores))

    to_drop = set()

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col_i = numeric_cols[i]
            col_j = numeric_cols[j]

            if col_i in to_drop or col_j in to_drop:
                continue

            corr_value = corr_matrix.loc[col_i, col_j]

            if corr_value >= threshold:
                mi_i = mi_dict.get(col_i, 0.0)
                mi_j = mi_dict.get(col_j, 0.0)

                if mi_i < mi_j:
                    to_drop.add(col_i)
                elif mi_j < mi_i:
                    to_drop.add(col_j)
                else:
                    if len(col_i) > len(col_j):
                        to_drop.add(col_i)
                    else:
                        to_drop.add(col_j)

    return sorted(to_drop)


# Baseline Importance
def get_baseline_features_importance(
        X: pd.DataFrame,
        Y: pd.Series):
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X,
        Y,
        test_size = test_size,
        stratify = Y,
        random_state = random_state
    )

    model = RandomForestClassifier(
        n_estimators = 300,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        random_state = random_state,
        n_jobs = 1,
        class_weight = "balanced"
    )

    model.fit(X_train, Y_train)

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_
        }
    ).sort_values(by = "importance", ascending = False, ignore_index = True)

    return importance_df


# Finalize Selected Features
def finalize_features(
    X: pd.DataFrame,
    importance_df: pd.DataFrame,
    importance_threshold: float = importance_threshold):
    selected = importance_df.loc[
        importance_df["importance"] >= importance_threshold, "feature"
    ].tolist()

    if len(selected) == 0:
        selected = importance_df["feature"].head(20).tolist()

    return [col for col in X.columns if col in selected]


# Save Outputs
def save_final_features(features: list[str], output_file: Path):
    output_file.write_text("\n".join(features), encoding="utf-8")

def save_selection_summary(
    constant_features: list[str],
    near_constant_features: list[str],
    correlated_features: list[str],
    importance_df: pd.DataFrame,
    final_features: list[str],
    output_file: Path,):
    lines = [
        "# Feature Selection Summary",
        "",
        "## Removed Constant Features",
        "",
    ]

    if constant_features:
        lines.extend([f"- {col}" for col in constant_features])
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Removed Near-Constant Features",
        "",
    ])

    if near_constant_features:
        lines.extend([f"- {col}" for col in near_constant_features])
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Removed Highly Correlated Features",
        "",
    ])

    if correlated_features:
        lines.extend([f"- {col}" for col in correlated_features])
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Baseline Feature Importance (Top 20)",
        "",
        "| Feature | Importance |",
        "|---|---:|",
    ])

    for _, row in importance_df.head(20).iterrows():
        lines.append(f"| {row['feature']} | {row['importance']:.6f} |")

    lines.extend([
        "",
        "## Final Selected Features",
        "",
    ])

    lines.extend([f"- {col}" for col in final_features])

    output_file.write_text("\n".join(lines), encoding="utf-8")


# Main Pipeline
def main():
    df = load_data(input_file)
    X, Y = split_features_target(df)
    print("=" * 60)
    print("Loaded engineered dataset")
    print(f"Input shape: {df.shape}")
    print(f"Feature matrix shape before selection: {X.shape}")
    print("=" * 60)

    constant_features = get_constant_features(X)
    X = X.drop(columns = constant_features, errors = "ignore")

    near_constant_features = get_near_constant_features(X)
    X = X.drop(columns=near_constant_features, errors="ignore")

    correlated_features = get_correlated_feature_drops(X, Y)
    X = X.drop(columns = correlated_features, errors = "ignore")

    print("After deterministic filtering:")
    print(f"Remaining features: {X.shape[1]}")

    importance_df = get_baseline_features_importance(X, Y)

    final_features = finalize_features(X, importance_df)

    X_final = X[final_features].copy()
    df_final = pd.concat([X_final, Y], axis=1)

    save_final_features(final_features, final_features_file)
    save_selection_summary(
        constant_features=constant_features,
        near_constant_features=near_constant_features,
        correlated_features=correlated_features,
        importance_df=importance_df,
        final_features=final_features,
        output_file=selection_summary_file,
    )
    df_final.to_csv(output_file, index=False)

    print("=" * 60)
    print("Feature selection complete")
    print(f"Final selected feature count: {len(final_features)}")
    print(f"Selected dataset saved to: {output_file}")
    print(f"Final feature list saved to: {final_features_file}")
    print(f"Selection summary saved to: {selection_summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()