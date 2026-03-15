"""
download_data.py

# Goal
Get the dataset into data/raw/ exactly as-is (no edits) and log basic stats.

# Inputs
- Source path / kaggle / URL (decide one)

# Outputs
- data/raw/<dataset_file>
- reports/data_quality/raw_load_log.md

# Steps (fill code in order)
1) Define RAW_DIR and REPORT_DIR constants.
2) Create folders if they don't exist.
3) Download or copy the dataset to RAW_DIR.
4) Assert file exists and is non-empty.
5) Load file into a DataFrame.
6) Log:
   - shape
   - columns
   - dtypes
   - missing value counts
   - duplicate rows
   - class distribution (target)
7) Save a markdown log file with these results.
"""
"""Raw Data Loading and Validation"""
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

"""Define Project Paths"""
project_root = Path.cwd()
data_dir = project_root/"data"
raw_dir = data_dir/"raw"
reports_dir = project_root/"reports"

source_file = project_root/"dataset"/"cover_type.csv"
raw_file = raw_dir/"forest_cover.csv"
log_file = reports_dir/"raw_load_log.md"

"""Ensure Directories"""
def ensure_dirs():
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

"""Create a Copy of the data in raw"""
def copy_to_raw(source_path: Path, destination_path: Path):
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    shutil.copy2(source_path, destination_path)

"""Validate file exists and size"""
def validate_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Raw File Does Not Exist: {file_path}")

    size_bytes = file_path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"Raw File is Empty: {file_path}")

"""Validate file is readable"""
def load_dataframe(file_path: Path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"File exists but could not be read by Pandas: {e}") from e
    return df

"""Build Markdown log"""
def build_markdown_log(df: pd.DataFrame, file_path: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_size_kb = round(file_path.stat().st_size / 1024, 2)

    head_md = df.head().to_markdown(index = False)
    dtypes_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values
    })
    dtypes_md = dtypes_df.to_markdown(index = False)

    content = f"""# Raw Load Log
                
    **Timestamp:** {now}
    **Raw file:** `{file_path}`
    **File size (KB):** {file_size_kb}

    ## Dataset Shape
    - Rows: {df.shape[0]}
    - Columns: {df.shape[1]}
    
    ## First 5 Rows
    {head_md}

    ## Column Data Types
    {dtypes_md}
    """

    return content


"""MAIN EXECUTION"""
def main():
    ensure_dirs()
    copy_to_raw(source_file, raw_file)
    validate_file(raw_file)

    df = load_dataframe(raw_file)

    print("\nDataset loaded successfully.")
    print(f"Shape: {df.shape}\n")
    print("Head:")
    print(df.head(), "\n")
    print("Dtypes:")
    print(df.dtypes)

    log_text = build_markdown_log(df, raw_file)
    log_file.write_text(log_text, encoding = "utf-8")

    print(f"\nRaw load log saved to: {log_file}")

if __name__ == "main":
    main()