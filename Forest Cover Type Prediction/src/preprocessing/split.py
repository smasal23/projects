"""
split.py

# Goal
Create reproducible splits (train/test or train/val/test) with stratification.

# Inputs
- data/interim/dataset_cleaned.*

# Outputs
- data/processed/X_train.*
- data/processed/X_test.*
- data/processed/y_train.*
- data/processed/y_test.*
- reports/data_quality/split_summary.md

# Steps
1) Load cleaned dataset.
2) Separate X and y.
3) Train/test split:
   - define test_size
   - stratify=y
   - random_state=seed
4) (Optional) train/val split from train.
5) Verify:
   - no index overlap
   - class distribution preserved across splits
6) Save X/y splits.
7) Save split summary report.
"""

# 1) imports + paths + constants

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path.cwd()

input_file = project_root/"data"/"interim"/"forest_cover_cleaned.csv"
processed_dir = project_root/"data"/"processed"
docs_dir = project_root/"docs"

X_train_file = processed_dir/"X_train.csv"
X_test_file = processed_dir/"X_test.csv"
Y_train_file = processed_dir/"Y_train.csv"
Y_test_file = processed_dir/"Y_test.csv"

summary_file = docs_dir/"split_summary.md"

target_col = "cover_type"
test_size = 0.20
random_state = 42


# 2) load cleaned
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


# 3) split logic
def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns = [target_col]).copy()
    Y = df[target_col].copy()
    return X, Y

# Perform Stratified Train/Test Split to split data into train and test sets while preserving target class proportions.
def perform_train_test_split(X: pd.DataFrame, Y: pd.Series, test_size: float, random_state: int):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state, stratify=Y)
    return X_train, X_test, Y_train, Y_test


# 4) checks (overlap, distributions)
def get_class_distribution(Y: pd.Series):
    counts = Y.value_counts().sort_index()
    proportions = Y.value_counts(normalize = True).sort_index() * 100

    summary = pd.DataFrame({
        "count": counts,
        "percentage": proportions.round(2)
    })

    summary.index.name = "cover_type"
    return summary


def check_no_overlap(X_train: pd.DataFrame, X_test: pd.DataFrame):
    train_index_set = set(X_train.index)
    test_index_set = set(X_test.index)

    overlap = train_index_set.intersection(test_index_set)

    return {
        "train_rows": len(train_index_set),
        "test_rows": len(test_index_set),
        "overlap_count": len(overlap),
        "overlap_found": len(overlap) > 0
    }


# 5) save splits
def save_splits(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.Series, Y_test: pd.Series):
    X_train.to_csv(X_train_file, index = False)
    X_test.to_csv(X_test_file, index = False)
    Y_train.to_frame(name = target_col).to_csv(Y_train_file, index = False)
    Y_test.to_frame(name = target_col).to_csv(Y_test_file, index = False)


# 6) write report
def build_split_summary(
    timestamp: str,
    input_file: Path,
    test_size: float,
    random_state: int,
    rows_total: int,
    cols_total: int,
    X_train_shape: tuple[int, int],
    X_test_shape: tuple[int, int],
    Y_train_shape: tuple[int],
    Y_test_shape: tuple[int],
    overlap_info: dict,
    full_dist: pd.DataFrame,
    train_dist: pd.DataFrame,
    test_dist: pd.DataFrame,
    ):
    report = f"""# Data Split Summary

    **Generated on:** {timestamp}
    
    ## Files
    - Input dataset: `{input_file}`
    - Saved outputs:
      - `data/processed/X_train.csv`
      - `data/processed/X_test.csv`
      - `data/processed/y_train.csv`
      - `data/processed/y_test.csv`
    
    ## Split Configuration
    - Target column: `{target_col}`
    - Split type: **Train/Test**
    - Test size: **{test_size:.0%}**
    - Train size: **{1 - test_size:.0%}**
    - Random state: **{random_state}**
    - Stratification: **Enabled**
    
    ## Dataset Shape Before Split
    - Total rows: **{rows_total}**
    - Total columns: **{cols_total}**
    
    ## Output Shapes
    - `X_train`: **{X_train_shape[0]} rows × {X_train_shape[1]} columns**
    - `X_test`: **{X_test_shape[0]} rows × {X_test_shape[1]} columns**
    - `y_train`: **{Y_train_shape[0]} rows**
    - `y_test`: **{Y_test_shape[0]} rows**
    
    ## No Overlap Check
    - Train rows checked: **{overlap_info['train_rows']}**
    - Test rows checked: **{overlap_info['test_rows']}**
    - Overlap count: **{overlap_info['overlap_count']}**
    - Overlap found: **{overlap_info['overlap_found']}**
    
    ## Full Dataset Target Distribution
    {full_dist.to_markdown()}
    
    ## Training Set Target Distribution
    {train_dist.to_markdown()}
    
    ## Test Set Target Distribution
    {test_dist.to_markdown()}
    
    ## Interpretation
    - Stratified splitting was applied using `{target_col}`.
    - This helps preserve class proportions in both training and test sets.
    - No overlap should exist between training and test samples.
    - The saved split files are ready for feature engineering and model training.
    """
    return report

def write_summary(report_text: str, path: Path):
    path.write_text(report_text, encoding="utf-8")


# Main Pipeline
def main():
    df = load_data(input_file)

    rows_total, cols_total = df.shape

    X, Y = split_features_target(df, target_col)

    X_train, X_test, Y_train, Y_test = perform_train_test_split(X = X, Y = Y, test_size = test_size, random_state = random_state)

    full_dist = get_class_distribution(Y)
    train_dist = get_class_distribution(Y_train)
    test_dist = get_class_distribution(Y_test)

    overlap_info = check_no_overlap(X_train, X_test)

    save_splits(X_train, X_test, Y_train, Y_test)

    report_text = build_split_summary(
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        input_file = input_file,
        test_size = test_size,
        random_state = random_state,
        rows_total = rows_total,
        cols_total = cols_total,
        X_train_shape = X_train.shape,
        X_test_shape = X_test.shape,
        Y_train_shape = Y_train.shape,
        Y_test_shape = Y_test.shape,
        overlap_info = overlap_info,
        full_dist = full_dist,
        train_dist = train_dist,
        test_dist = test_dist,
    )

    write_summary(report_text, summary_file)

    print("Split completed successfully.")
    print(f"X_train saved to: {X_train_file}")
    print(f"X_test saved to: {X_test_file}")
    print(f"y_train saved to: {Y_train_file}")
    print(f"y_test saved to: {Y_test_file}")
    print(f"Split summary saved to: {summary_file}")
    print(f"Train shape: {X_train.shape}, {Y_train.shape}")
    print(f"Test shape: {X_test.shape}, {Y_test.shape}")
    print("Overlap count:", overlap_info["overlap_count"])


if __name__ == "__main__":
    main()