"""
train.py

# Goal
Train baseline model pipelines and save results.

# Inputs
- data/processed/X_train, y_train
- preprocessor from src/features/encoders.py
- model from src/modeling/models.py

# Outputs
- reports/model_results/baseline_scores.csv (or json)
- optional: saved baseline pipeline artifact

# Steps
1) Load train split.
2) Build preprocessing transformer.
3) Choose baseline models list.
4) For each model:
   - build Pipeline(preprocessor + model)
   - cross-validate using StratifiedKFold
   - compute mean/std macro F1 (and accuracy)
   - log results
x5) Save scores table.
6) Pick best baseline candidate and save it (optional).
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from joblib import dump

from src.data.download_data import project_root
from src.features.selection import load_data, split_features_target
from src.features.encoders import build_preprocessor
from src.modeling.models import get_baseline_models
from src.modeling.metrics import evaluate_classification, save_confusion_matrix_png, save_evaluation_outputs


# Paths + Configs
project_root = Path(__file__).resolve().parents[2]

config_dir = project_root/"config"
paths_file = config_dir/"paths.yaml"
train_file = config_dir/"train.yaml"

with open(paths_file, "r", encoding = "utf-8") as f:
    paths_cfg = yaml.safe_load(f)

with open(train_file, "r", encoding = "utf-8") as f:
    train_cfg = yaml.safe_load(f)

data_dir = project_root/paths_cfg["data"]["processed_dir"]
reports_dir = project_root/paths_cfg["artifacts"]["reports_dir"]
figures_dir = project_root/paths_cfg["artifacts"]["figures_dir"]
models_dir = project_root/paths_cfg["artifacts"]["models_dir"]

reports_dir.mkdir(parents = True, exist_ok = True)
figures_dir.mkdir(parents = True, exist_ok = True)
models_dir.mkdir(parents = True, exist_ok = True)

input_file = data_dir/"forest_cover_selected.csv"

target_col = train_cfg["data"]["target_col"]
drop_cols = train_cfg["data"].get("drop_cols", [])
id_cols = train_cfg["data"].get("id_cols", [])

test_size = train_cfg["split"]["test_size"]
random_state = train_cfg["split"]["random_state"]
stratify = train_cfg["split"]["stratify"]
shuffle = train_cfg["split"]["shuffle"]

save_model = train_cfg["artifacts"]["save_model"]
model_path = project_root/train_cfg["artifacts"]["model_path"]
metrics_path = project_root/train_cfg["artifacts"]["metrics_path"]
classification_report_path = project_root/train_cfg["artifacts"]["classification_report_path"]

classification_report_csv_path = reports_dir/"classification_report.csv"
confusion_matrix_csv_path = reports_dir/"confusion_matrix.csv"
confusion_matrix_png_path = figures_dir/"confusion_matrix.png"
baseline_results_csv_path = reports_dir/"baseline_results.csv"


# Utility functions
def split_feature_types(df: pd.DataFrame, target_col: str):
    feature_cols = [col for col in df.columns if col != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    return numeric_cols, categorical_cols

def prepare_modeling_dataframe(
        df: pd.DataFrame,
        target_col: str,
        drop_cols: list[str] | None = None,
        id_cols: list[str] | None = None
):
    drop_cols = drop_cols or []
    id_cols = id_cols or []

    removable_cols = [col for col in drop_cols + id_cols if col in df.columns]
    return df.drop(columns = removable_cols, errors = "ignore").copy()

def build_model_pipeline(
        numeric_cols: list[str],
        categorical_cols: list[str],
        model
):
    preprocessor = build_preprocessor(numeric_cols = numeric_cols, categorical_cols = categorical_cols)

    pipeline = Pipeline(steps = [("preprocessor", preprocessor), ("model", model)])

    return pipeline

def save_baseline_results(results: list[dict], output_path: Path):
    results_df = pd.DataFrame(results)

    if "cv_f1_macro_mean" in results_df.columns:
        sort_cols = ["cv_f1_macro_mean", "cv_accuracy_mean"]
    else:
        sort_cols = ["f1_macro", "accuracy"]

    results_df = results_df.sort_values(
        by=sort_cols,
        ascending=False,
        ignore_index=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return output_path


# Main Training Flow
def main():
    df = load_data(input_file)
    df= prepare_modeling_dataframe(df= df, target_col = target_col, drop_cols = drop_cols, id_cols = id_cols)

    X, y = split_features_target(df, target_col = target_col)

    print("=" * 70)
    print("Loaded selected dataset")
    print(f"Input file: {input_file}")
    print(f"Shape after optional column removal: {df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print("=" * 70)

    numeric_cols, categorical_cols = split_feature_types(df = df, target_col = target_col)

    print(f"Target column: {target_col}")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = random_state,
        stratify = y if stratify else None,
        shuffle = shuffle
     )

    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_train.to_frame(name=target_col).to_csv(data_dir / "y_train.csv", index=False)
    y_test.to_frame(name=target_col).to_csv(data_dir / "y_test.csv", index=False)

    print("-" * 70)
    print("Train/test split complete")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")
    print("-" * 70)

    models = get_baseline_models(random_state = random_state)
    baseline_results: list[dict] = []

    best_model_name = None
    best_pipeline = None
    best_f1_macro = -1.0
    best_evaluation = None
    best_y_pred = None

    for model_name, model in models.items():
        print(f"\n Training model: {model_name}")

        pipeline = build_model_pipeline(numeric_cols = numeric_cols, categorical_cols = categorical_cols, model = model)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        evaluation = evaluate_classification(y_test, y_pred)
        metrics = evaluation["metrics"]

        baseline_results.append(
            {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "n_features_in": X_train.shape[1],
                "n_train_rows": X_train.shape[0],
                "n_test_rows": X_test.shape[0]
            }
        )

        print(
            f"accuracy = {metrics["accuracy"]:.4f} | "
            f"f1_macro = {metrics["f1_macro"]:.4f} | "
            f"f1_weighted = {metrics["f1_weighted"]:.4f} | "
        )

        if metrics["f1_macro"] > best_f1_macro:
            best_f1_macro = metrics["f1_macro"]
            best_model_name = model_name
            best_pipeline = pipeline
            best_evaluation = evaluation
            best_y_pred = y_pred

    baseline_results_path = save_baseline_results(results = baseline_results, output_path = baseline_results_csv_path)

    print("\n" + "=" * 70)
    print("Baseline comparison complete")
    print(f"Best model: {best_model_name}")
    print(f"Best macro F1: {best_f1_macro:.4f}")
    print(f"Baseline results saved to: {baseline_results_path}")
    print("=" * 70)

    if best_evaluation is None or best_pipeline is None or best_y_pred is None:
        raise RuntimeError("No valid best model was found during training.")

    save_evaluation_outputs(
        evaluation = best_evaluation,
        metrics_json_path = metrics_path,
        classification_report_csv_path = classification_report_csv_path,
        confusion_matrix_csv_path = confusion_matrix_csv_path,
    )

    save_confusion_matrix_png(
        y_true=y_test,
        y_pred=best_y_pred,
        output_path=confusion_matrix_png_path,
        title=f"Confusion Matrix - {best_model_name}",
    )

    if save_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(best_pipeline, model_path)
        print(f"Best pipeline saved to: {model_path}")

if __name__ == "__main__":
    main()
