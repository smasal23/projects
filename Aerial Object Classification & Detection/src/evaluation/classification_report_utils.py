from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.utils.io import save_dataframe, save_json, save_text


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas-heavy objects into JSON-serializable forms.
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
        return obj.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    return obj


def classification_report_dict_to_dataframe(report_dict: Dict) -> pd.DataFrame:
    """
    Convert sklearn-style classification_report(output_dict=True) payload
    into a tidy dataframe.
    """
    rows = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            row = {"label": label}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        preferred_cols = [
            "label",
            "precision",
            "recall",
            "f1-score",
            "support",
        ]
        ordered_cols = [col for col in preferred_cols if col in df.columns] + [
            col for col in df.columns if col not in preferred_cols
        ]
        df = df[ordered_cols]

    return df


def build_metrics_summary_markdown(
    model_name: str,
    metrics_payload: Dict,
    confusion_matrix_path: str | None = None,
    curves_paths: Sequence[str] | None = None,
) -> str:
    """
    Build a compact markdown summary for one evaluated classifier.
    """
    curves_paths = list(curves_paths or [])
    lines: List[str] = [
        f"# Evaluation Summary — {model_name}",
        "",
        "## Core Test Metrics",
        "",
        f"- Accuracy: {metrics_payload['accuracy']:.4f}",
        f"- Precision: {metrics_payload['precision']:.4f}",
        f"- Recall: {metrics_payload['recall']:.4f}",
        f"- F1-score: {metrics_payload['f1_score']:.4f}",
    ]

    if metrics_payload.get("training_time_seconds") is not None:
        lines.append(f"- Training time (seconds): {metrics_payload['training_time_seconds']:.2f}")

    if metrics_payload.get("model_size_mb") is not None:
        lines.append(f"- Model size (MB): {metrics_payload['model_size_mb']:.3f}")

    if metrics_payload.get("generalization_gap_abs") is not None:
        lines.append(
            f"- |Validation Accuracy - Test Accuracy|: {metrics_payload['generalization_gap_abs']:.4f}"
        )

    if confusion_matrix_path:
        lines.extend([
            "",
            "## Confusion Matrix Artifact",
            "",
            f"- {confusion_matrix_path}",
        ])

    if curves_paths:
        lines.extend([
            "",
            "## Curves",
            "",
        ])
        for path in curves_paths:
            lines.append(f"- {path}")

    report_dict = metrics_payload.get("classification_report", {})
    if report_dict:
        report_df = classification_report_dict_to_dataframe(report_dict)
        if not report_df.empty:
            lines.extend([
                "",
                "## Classification Report",
                "",
                report_df.to_markdown(index=False),
            ])

    return "\n".join(lines)


def save_classification_artifacts(
    metrics_payload: Dict,
    metrics_json_path: Path,
    report_md_path: Path,
    classification_report_csv_path: Path | None = None,
) -> None:
    """
    Save the evaluation metrics JSON, markdown report, and optional report CSV.
    """
    serializable_payload = _make_json_serializable(metrics_payload)
    save_json(serializable_payload, metrics_json_path)

    markdown_text = build_metrics_summary_markdown(
        model_name=metrics_payload["model_name"],
        metrics_payload=metrics_payload,
        confusion_matrix_path=metrics_payload.get("confusion_matrix_path"),
        curves_paths=metrics_payload.get("curves_paths", []),
    )
    save_text(markdown_text, report_md_path)

    report_dict = metrics_payload.get("classification_report", {})
    if classification_report_csv_path is not None and report_dict:
        report_df = classification_report_dict_to_dataframe(report_dict)
        save_dataframe(report_df, classification_report_csv_path, index=False)