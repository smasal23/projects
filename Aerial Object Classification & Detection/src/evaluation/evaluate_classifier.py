from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from src.evaluation.compare_models import (
    build_model_comparison_table,
    save_model_comparison_barplot,
    save_model_comparison_csv,
    select_best_model,
)
from src.evaluation.error_analysis import (
    build_prediction_analysis_table,
    save_error_analysis_grid,
    split_binary_error_types,
    summarize_confusion_patterns,
)
from src.features.augmentations import build_eval_preprocessing_pipeline
from src.features.preprocessing_transforms import build_tf_transfer_preprocess_layer
from src.modeling.losses_metrics import evaluate_classification_predictions
from src.utils.io import save_json


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas/path objects into JSON-serializable forms.
    """
    if isinstance(obj, dict):
        return {str(key): _make_json_serializable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]

    if isinstance(obj, tuple):
        return [_make_json_serializable(item) for item in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, pd.DataFrame):
        return [_make_json_serializable(row) for row in obj.to_dict(orient="records")]

    if isinstance(obj, pd.Series):
        return _make_json_serializable(obj.to_dict())

    return obj


def evaluate_keras_classifier_on_paths(
    model_path: Path,
    image_paths: Sequence[Path],
    y_true: Sequence[int],
    class_names: Sequence[str],
    image_size: Sequence[int],
    preprocess_mode: str,
    backbone_name: str | None = None,
) -> Dict:
    """
    Evaluate one saved .keras model on a list of image paths.

    preprocess_mode:
      - 'custom'
      - 'transfer'
    """
    model_path = Path(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    if preprocess_mode == "custom":
        preprocess_layer = build_eval_preprocessing_pipeline()
    elif preprocess_mode == "transfer":
        if not backbone_name:
            raise ValueError("backbone_name is required when preprocess_mode='transfer'.")
        preprocess_layer = build_tf_transfer_preprocess_layer(backbone_name)
    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    y_pred: List[int] = []
    y_score: List[float] = []

    for image_path in image_paths:
        image = tf.keras.utils.load_img(image_path, target_size=tuple(image_size))
        image = tf.keras.utils.img_to_array(image)
        image = tf.expand_dims(image, axis=0)
        image = preprocess_layer(image, training=False)

        pred = model.predict(image, verbose=0)

        if pred.shape[-1] == 1:
            score = float(pred.reshape(-1)[0])
            pred_idx = int(score >= 0.5)
        else:
            pred_vec = pred.reshape(-1)
            pred_idx = int(np.argmax(pred_vec))
            score = float(np.max(pred_vec))

        y_pred.append(pred_idx)
        y_score.append(score)

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)

    metrics = evaluate_classification_predictions(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        class_names=class_names,
        label_mode="binary",
    )

    return {
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "y_score": y_score_arr,
        "metrics": metrics,
    }


def build_evaluated_model_payload(
    model_name: str,
    backbone: str,
    model_path: Path,
    saved_metrics_path: Path | None,
    eval_output: Dict,
    class_names: Sequence[str],
    val_accuracy: float | None = None,
    training_time_seconds: float | None = None,
    speed_bucket: str = "unknown",
    deployment_note: str = "Deployment suitability should be validated.",
) -> Dict:
    """
    Merge fresh evaluation metrics with saved metadata.
    """
    model_size_mb = round(Path(model_path).stat().st_size / (1024 * 1024), 3)
    test_accuracy = float(eval_output["metrics"]["accuracy"])

    generalization_gap_abs = None
    if val_accuracy is not None:
        generalization_gap_abs = abs(val_accuracy - test_accuracy)

    payload = {
        "model_name": model_name,
        "backbone": backbone,
        "model_path": str(model_path),
        "metrics_path": None if saved_metrics_path is None else str(saved_metrics_path),
        "accuracy": float(eval_output["metrics"]["accuracy"]),
        "precision": float(eval_output["metrics"]["precision"]),
        "recall": float(eval_output["metrics"]["recall"]),
        "f1_score": float(eval_output["metrics"]["f1_score"]),
        "val_accuracy": val_accuracy,
        "generalization_gap_abs": generalization_gap_abs,
        "training_time_seconds": training_time_seconds,
        "model_size_mb": model_size_mb,
        "speed_bucket": speed_bucket,
        "deployment_note": deployment_note,
        "confusion_matrix": eval_output["metrics"]["confusion_matrix"],
        "classification_report": eval_output["metrics"]["classification_report"],
        "class_names": list(class_names),
    }
    return payload


def export_final_selected_classifier(
    selected_row: pd.Series,
    final_model_path: Path,
    final_class_mapping_path: Path,
    final_metrics_path: Path,
    class_to_index: Dict[str, int],
    index_to_class: Dict[int, str],
    all_comparisons_df: pd.DataFrame,
) -> Dict:
    """
    Export selected model, class mapping, and final metrics JSON.
    """
    final_model_path = Path(final_model_path)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    source_model_path = Path(selected_row["model_path"])
    if source_model_path.resolve() != final_model_path.resolve():
        final_model_path.write_bytes(source_model_path.read_bytes())

    save_json(
        {
            "class_to_index": class_to_index,
            "index_to_class": {str(k): v for k, v in index_to_class.items()},
        },
        final_class_mapping_path,
    )

    comparison_export_df = all_comparisons_df.copy()

    drop_if_present = [
        "y_true",
        "y_pred",
        "y_score",
    ]
    comparison_export_df = comparison_export_df.drop(
        columns=[col for col in drop_if_present if col in comparison_export_df.columns],
        errors="ignore",
    )

    final_payload = {
        "selected_model_name": selected_row["model_name"],
        "selected_model_path": str(source_model_path),
        "exported_final_model_path": str(final_model_path),
        "class_mapping_path": str(final_class_mapping_path),
        "selected_model_metrics": {
            "accuracy": float(selected_row["accuracy"]),
            "precision": float(selected_row["precision"]),
            "recall": float(selected_row["recall"]),
            "f1_score": float(selected_row["f1_score"]),
            "val_accuracy": None if pd.isna(selected_row["val_accuracy"]) else float(selected_row["val_accuracy"]),
            "generalization_gap_abs": None if pd.isna(selected_row["generalization_gap_abs"]) else float(selected_row["generalization_gap_abs"]),
            "training_time_seconds": None if pd.isna(selected_row["training_time_seconds"]) else float(selected_row["training_time_seconds"]),
            "model_size_mb": float(selected_row["model_size_mb"]),
            "speed_bucket": selected_row["speed_bucket"],
            "deployment_note": selected_row["deployment_note"],
            "streamlit_suitability": selected_row["streamlit_suitability"],
        },
        "all_model_comparisons": comparison_export_df.to_dict(orient="records"),
    }

    final_payload = _make_json_serializable(final_payload)
    save_json(final_payload, final_metrics_path)
    return final_payload


def run_full_evaluation_suite(
    evaluated_model_payloads: List[Dict],
    image_paths: Sequence[Path],
    y_true: Sequence[int],
    class_names: Sequence[str],
    index_to_class: Dict[int, str],
    comparison_csv_path: Path,
    comparison_plot_path: Path,
    error_analysis_grid_path: Path,
) -> Dict:
    """
    Finalize comparison outputs and perform error analysis on the best model.
    """
    comparison_df = build_model_comparison_table(evaluated_model_payloads)

    save_model_comparison_csv(comparison_df, comparison_csv_path)
    save_model_comparison_barplot(comparison_df, comparison_plot_path)

    selected_row = select_best_model(comparison_df)
    selected_payload = next(
        row for row in evaluated_model_payloads if row["model_name"] == selected_row["model_name"]
    )

    prediction_df = build_prediction_analysis_table(
        image_paths=image_paths,
        y_true=y_true,
        y_pred=selected_payload["y_pred"],
        y_score=selected_payload["y_score"],
        index_to_class=index_to_class,
    )

    error_tables = split_binary_error_types(
        prediction_df=prediction_df,
        negative_class_name=list(class_names)[0],
        positive_class_name=list(class_names)[1],
    )

    error_summary = summarize_confusion_patterns(error_tables)
    save_error_analysis_grid(
        prediction_df=prediction_df,
        output_path=error_analysis_grid_path,
        title=f"Hard Error Analysis — {selected_row['model_name']}",
    )

    return {
        "comparison_df": comparison_df,
        "selected_row": selected_row,
        "prediction_df": prediction_df,
        "error_tables": error_tables,
        "error_summary": error_summary,
    }