from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import save_dataframe


def infer_streamlit_suitability(speed_bucket: str) -> str:
    mapping = {
        "fast": "high",
        "medium": "good",
        "slower": "limited",
        "unknown": "check",
    }
    return mapping.get(speed_bucket, "check")


def build_model_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Build a comparison table from evaluated model payloads.
    """
    df = pd.DataFrame(results).copy()
    if df.empty:
        return df

    if "speed_bucket" in df.columns:
        df["streamlit_suitability"] = df["speed_bucket"].map(infer_streamlit_suitability)

    if "generalization_gap_abs" not in df.columns:
        df["generalization_gap_abs"] = np.nan

    df["selection_score"] = (
        0.40 * df["f1_score"]
        + 0.25 * df["accuracy"]
        + 0.15 * df["precision"]
        + 0.10 * df["recall"]
        + 0.10 * (1 - df["generalization_gap_abs"].fillna(0.0))
    )

    df = df.sort_values(
        by=["selection_score", "f1_score", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return df


def save_model_comparison_csv(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Save comparison dataframe as CSV.
    """
    save_dataframe(comparison_df, output_path, index=False)
    return output_path


def save_model_comparison_barplot(
    comparison_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """
    Save grouped bar chart for accuracy/precision/recall/F1.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = comparison_df[["model_name", "accuracy", "precision", "recall", "f1_score"]].copy()
    metrics_order = ["accuracy", "precision", "recall", "f1_score"]
    model_names = plot_df["model_name"].tolist()

    x = np.arange(len(model_names))
    bar_width = 0.20

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric_name in enumerate(metrics_order):
        metric_values = plot_df[metric_name].values
        ax.bar(x + (i - 1.5) * bar_width, metric_values, width=bar_width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test-Set Metric Comparison Across Models")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def select_best_model(comparison_df: pd.DataFrame) -> pd.Series:
    """
    Select the top-ranked model row.
    """
    if comparison_df.empty:
        raise ValueError("comparison_df is empty.")
    return comparison_df.iloc[0].copy()