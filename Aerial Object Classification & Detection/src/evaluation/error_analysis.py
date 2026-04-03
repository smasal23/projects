from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def build_prediction_analysis_table(
    image_paths: Sequence[Path],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    index_to_class: Dict[int, str],
) -> pd.DataFrame:
    """
    Build a detailed per-image prediction table for downstream error analysis.
    """
    df = pd.DataFrame(
        {
            "image_path": [str(Path(p)) for p in image_paths],
            "true_label_idx": list(y_true),
            "pred_label_idx": list(y_pred),
            "pred_score": list(y_score),
        }
    )

    df["true_label"] = df["true_label_idx"].map(index_to_class)
    df["pred_label"] = df["pred_label_idx"].map(index_to_class)
    df["is_correct"] = df["true_label_idx"] == df["pred_label_idx"]

    return df


def split_binary_error_types(
    prediction_df: pd.DataFrame,
    negative_class_name: str,
    positive_class_name: str,
) -> Dict[str, pd.DataFrame]:
    """
    Split misclassifications into false positives and false negatives for binary classification.
    """
    false_positives = prediction_df[
        (prediction_df["true_label"] == negative_class_name) &
        (prediction_df["pred_label"] == positive_class_name)
    ].copy()

    false_negatives = prediction_df[
        (prediction_df["true_label"] == positive_class_name) &
        (prediction_df["pred_label"] == negative_class_name)
    ].copy()

    hard_images = prediction_df.loc[~prediction_df["is_correct"]].copy()
    hard_images = hard_images.sort_values(by="pred_score", ascending=False).reset_index(drop=True)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hard_images": hard_images,
    }


def summarize_confusion_patterns(error_tables: Dict[str, pd.DataFrame]) -> str:
    """
    Produce a concise text summary of binary-class confusion patterns.
    """
    fp_count = len(error_tables["false_positives"])
    fn_count = len(error_tables["false_negatives"])
    hard_count = len(error_tables["hard_images"])

    dominant = "balanced"
    if fp_count > fn_count:
        dominant = "more false positives than false negatives"
    elif fn_count > fp_count:
        dominant = "more false negatives than false positives"

    return (
        f"False positives: {fp_count}, false negatives: {fn_count}, total hard images: {hard_count}. "
        f"The current error distribution shows {dominant}."
    )


def save_error_analysis_grid(
    prediction_df: pd.DataFrame,
    output_path: Path,
    title: str,
    max_images: int = 12,
    ncols: int = 4,
    dpi: int = 150,
) -> Path:
    """
    Save a grid of the hardest misclassified images ranked by confidence.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_df = prediction_df.loc[~prediction_df["is_correct"]].copy()
    error_df = error_df.sort_values(by="pred_score", ascending=False).head(max_images)

    if error_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No misclassified images found.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path

    n_images = len(error_df)
    nrows = math.ceil(n_images / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, error_df.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"T: {row['true_label']} | P: {row['pred_label']}\nScore: {row['pred_score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path