"""
make_dataset.py

# Goal
Create an initial "interim" dataset from raw data (standardize column names only),
and prepare data dictionary draft.

# Inputs
- data/raw/<dataset_file>

# Outputs
- data/interim/dataset_interim.parquet (or .csv)
- reports/data_quality/data_dictionary_draft.md

# Steps
1) Load raw dataset.
2) Standardize column names (lowercase, underscores).
3) Ensure target column exists.
4) Compute class counts & percentages.
5) Save interim dataset (no feature engineering here).
6) Generate a basic data dictionary:
   - column name
   - dtype
   - min/max (numeric)
   - unique counts (categorical)
7) Save data dictionary draft in reports/.
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================

from pathlib import Path
import re
import pandas as pd

# =========================
# 2. DEFINE PROJECT PATHS
# =========================

project_root = Path.cwd()
raw_file = project_root/"data"/"raw"/"forest_cover.csv"
interim_dir = project_root/"data"/"interim"
docs_dir = project_root/"docs"

interim_file = interim_dir/"forest_cover_interim.csv"
data_dict_file = docs_dir/"data_dictionary_draft.md"

target_candidates = ["cover_type"]

# =========================
# 3. CREATE REQUIRED FOLDERS
# =========================
def ensure_dirs():
    interim_dir.mkdir(parents = True, exist_ok = True)
    docs_dir.mkdir(parents = True, exist_ok = True)

# =========================
# 4. STANDARDIZE COLUMN NAMES
# =========================
def standardize_column_names(col: str):
    col = col.strip().lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")

def standardize_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [standardize_column_names(col) for col in df.columns]
    return df

# =========================
# 5. VALIDATE TARGET COLUMN
# =========================

def validate_target(df: pd.DataFrame):
    for target in target_candidates:
        if target in df.columns:
            return target

    raise ValueError(
        f"Target column not found. Expected one of: {target_candidates}."
        f"Available columns: {list(df.columns)}"
    )

# =========================
# 6. BUILD DRAFT DATA DICTIONARY
# =========================

def build_data_dictionary(df: pd.DataFrame, target_col: str):
    rows = []

    for col in df.columns:
        rows.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "is_target": "yes" if col == target_col else "no",
            "null_count": int(df[col].isna().sum()),
            "example_value": str(df[col].iloc[0]) if len(df) > 0 else ""
        })

    dict_df = pd.DataFrame(rows)
    dict_md = dict_df.to_markdown(index = False)

    content = f"""# Draft Data Dictionary
    
    This is the first draft of the data dictionary generated from the interim dataset.

    ## Dataset Summary
    - Rows: {df.shape[0]}
    - Columns: {df.shape[1]}
    - Target Column: `{target_col}`
    
    ## Columns
    {dict_md}

    ## Notes
    - Descriptions will be refined later using the official project brief.
    - Units/Ranges and derived features can be added in later phases.
    """

    return content

# =========================
# 7. MAIN WORKFLOW
# =========================
def main():
    ensure_dirs()

    # Validate raw file existence
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")

    # Load raw dataset
    df = pd.read_csv(raw_file)

    # Standardize column names
    df = standardize_columns(df)

    # Validate target columns
    target_col = validate_target(df)

    # Save first interim dataset
    df.to_csv(interim_file, index = False)
    print(f"Interim dataset saved to: {interim_file}")

    # Create and save draft data dictionary
    data_dict_text = build_data_dictionary(df, target_col)
    data_dict_file.write_text(data_dict_text, encoding = "utf-8")
    print(f"Draft data dictionary saved to: {data_dict_file}")


if __name__ == "__main__":
    main()
