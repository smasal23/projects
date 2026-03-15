"""
evaluate.py

# Goal
Evaluate a chosen trained pipeline on the test set and save artifacts.

# Inputs
- models/best_model.joblib
- data/processed/X_test, y_test

# Outputs
- reports/model_results/final_metrics.json
- reports/model_results/confusion_matrix.png
- reports/model_results/classification_report.txt

# Steps
1) Load model artifact.
2) Load test split.
3) Predict y_pred.
4) Compute metrics + report + confusion matrix.
5) Save all artifacts.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from joblib import load

from src.modeling.metrics import (
    evaluate_classification,
    save_confusion_matrix_png,
    save_evaluation_outputs,
)


PROJECT_ROOT = Path.cwd()

CONFIG_DIR = PROJECT_ROOT / "config"
PATHS_FILE = CONFIG_DIR / "paths.yaml"
TRAIN_FILE = CONFIG_DIR / "train.yaml"

with open(PATHS_FILE, "r", encoding="utf-8") as f:
    paths_cfg = yaml.safe_load(f)

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_cfg = yaml.safe_load(f)


DATA_DIR = PROJECT_ROOT / paths_cfg["data"]["processed_dir"]
REPORTS_DIR = PROJECT_ROOT / paths_cfg["artifacts"]["reports_dir"]
FIGURES_DIR = PROJECT_ROOT / paths_cfg["artifacts"]["figures_dir"]
MODELS_DIR = PROJECT_ROOT / paths_cfg["artifacts"]["models_dir"]

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_PATH = PROJECT_ROOT / train_cfg["artifacts"]["model_path"]
TARGET_COL = train_cfg["data"]["target_col"]

TEST_FEATURES_FILE = DATA_DIR / "X_test.csv"
TEST_LABELS_FILE = DATA_DIR / "y_test.csv"

METRICS_JSON_PATH = PROJECT_ROOT / train_cfg["artifacts"]["metrics_path"]
CLASSIFICATION_REPORT_TXT_PATH = (
    PROJECT_ROOT / train_cfg["artifacts"]["classification_report_path"]
)

CLASSIFICATION_REPORT_CSV_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG_PATH = FIGURES_DIR / "confusion_matrix.png"


def load_pipeline(model_path: Path):
    """
    Load the saved pipeline artifact.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model pipeline not found: {model_path}")
    return load(model_path)


def load_test_data(
    x_test_file: Path,
    y_test_file: Path,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load held-out test features and labels.
    """
    if not x_test_file.exists():
        raise FileNotFoundError(f"Test features file not found: {x_test_file}")

    if not y_test_file.exists():
        raise FileNotFoundError(f"Test labels file not found: {y_test_file}")

    X_test = pd.read_csv(x_test_file)
    y_test_df = pd.read_csv(y_test_file)

    if target_col not in y_test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in labels file: {y_test_file}"
        )

    y_test = y_test_df[target_col].copy()
    return X_test, y_test


def main() -> None:
    print("=" * 70)
    print("Loading saved pipeline artifact")
    print("=" * 70)

    pipeline = load_pipeline(MODEL_PATH)
    print(f"Loaded pipeline from: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("Loading held-out test data")
    print("=" * 70)

    X_test, y_test = load_test_data(
        x_test_file=TEST_FEATURES_FILE,
        y_test_file=TEST_LABELS_FILE,
        target_col=TARGET_COL,
    )

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\n" + "=" * 70)
    print("Generating predictions")
    print("=" * 70)

    y_pred = pipeline.predict(X_test)

    evaluation = evaluate_classification(y_test, y_pred)

    print("\nFinal metrics:")
    for metric_name, metric_value in evaluation["metrics"].items():
        print(f"{metric_name}: {metric_value:.4f}")

    save_evaluation_outputs(
        evaluation=evaluation,
        metrics_json_path=METRICS_JSON_PATH,
        classification_report_csv_path=CLASSIFICATION_REPORT_CSV_PATH,
        confusion_matrix_csv_path=CONFUSION_MATRIX_CSV_PATH,
    )

    save_confusion_matrix_png(
        y_true=y_test,
        y_pred=y_pred,
        output_path=CONFUSION_MATRIX_PNG_PATH,
        title="Final Confusion Matrix",
    )

    print("\n" + "=" * 70)
    print("Final evaluation artifacts saved")
    print(f"Metrics JSON: {METRICS_JSON_PATH}")
    print(f"Classification report CSV: {CLASSIFICATION_REPORT_CSV_PATH}")
    print(f"Confusion matrix CSV: {CONFUSION_MATRIX_CSV_PATH}")
    print(f"Confusion matrix PNG: {CONFUSION_MATRIX_PNG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()