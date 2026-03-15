"""
metrics.py

# Goal
Standardize evaluation outputs for classification.

# Outputs
- metrics dict
- classification report text
- confusion matrix
- optional: plots saved to reports/

# Steps
1) compute_metrics(y_true, y_pred) → dict
2) compute_report(y_true, y_pred) → str
3) compute_confusion(y_true, y_pred) → np.array
4) save_metrics_json(path, metrics)
5) save_confusion_matrix_plot(path, cm, labels)
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import tight_layout

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def ensure_parent_dir(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# def compute_metrics(...): ...
def compute_basic_metrics(y_true, y_pred):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average = "macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average = "weighted"))
    }
    return metrics


def get_confusion_matrix_df(y_true, y_pred, labels: list[Any] | None = None):
    if labels is None:
        labels = sorted(pd.Series(y_true).unique().tolist())

    cm = confusion_matrix(y_true, y_pred, labels = labels)

    cm_df = pd.DataFrame(
        cm,
        index = [f"true_{label}" for label in labels],
        columns = [f"pred_{label}" for label in labels]
    )
    return cm_df


def get_classification_report_dict(y_true, y_pred, labels: list | None = None):
    if labels is None:
        labels = sorted(pd.Series(y_true).unique().tolist())

    report_dict = classification_report(
        y_true,
        y_pred,
        labels = labels,
        output_dict = True,
        zero_division = 0
    )
    return report_dict


def get_classification_report_df(y_true, y_pred, labels: list[Any] | None = None):
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        labels=labels,
        zero_division=0
    )
    print(type(report_dict))
    print(report_dict)

    report_df = pd.DataFrame(report_dict).transpose()
    return report_df


def plot_confusion_matrix(y_true, y_pred, labels: list[Any] | None = None, title: str = "Confusion Matrix"):
    if labels is None:
        labels = sorted(pd.Series(y_true).unique().tolist())

    cm = confusion_matrix(y_true, y_pred, labels = labels)

    fig, ax = plt.subplots(figsize = (8, 6))
    im = ax.imshow(cm, interpolation = "nearest")
    fig.colorbar = (im)

    ax.set(
        xticks = np.arange(len(labels)),
        yticks = np.arange(len(labels)),
        xticklabels = labels,
        yticklabels = labels,
        xlabel = "Predicted Label",
        ylabel = "True Label",
        title = title
    )

    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

    threshold = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha = "center",
                va = "center",
                color = "white" if cm[i, j] > threshold else 'black'
            )
    fig,tight_layout()

    return fig


# def save_confusion_matrix_plot(...): ...
def save_metrics_json(metrics: dict[str, Any], output_path: str | Path):
    output_path = ensure_parent_dir(output_path)
    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(metrics, f, indent = 2, ensure_ascii = False)
    return output_path


def save_dataframe_csv(df: pd.DataFrame, output_path: str | Path):
    output_path = ensure_parent_dir(output_path)
    df.to_csv(output_path, index = True)
    return output_path


def save_confusion_matrix_png(y_true, y_pred, output_path: str | Path, labels: list[Any] | None = None, title: str = "Confusion Matrix"):
    output_path = ensure_parent_dir(output_path)
    fig = plot_confusion_matrix(
        y_true = y_true,
        y_pred = y_pred,
        labels = labels,
        title = title
    )
    fig.savefig(output_path, bbox_inches = "tight")
    plt.close(fig)

    return output_path

def evaluate_classification(
    y_true,
    y_pred,
    labels: list[Any] | None = None,
):
    basic_metrics = compute_basic_metrics(y_true, y_pred)
    cm_df = get_confusion_matrix_df(y_true, y_pred, labels=labels)
    report_df = get_classification_report_df(y_true, y_pred, labels=labels)
    report_dict = get_classification_report_dict(y_true, y_pred, labels=labels)

    return {
        "metrics": basic_metrics,
        "confusion_matrix_df": cm_df,
        "classification_report_df": report_df,
        "classification_report_dict": report_dict,
    }


def save_evaluation_outputs(
        evaluation: dict[str, Any],
        metrics_json_path: str | Path,
        classification_report_csv_path: str | Path,
        confusion_matrix_csv_path: str | Path
):
    saved_paths = {
        "metrics_json": save_metrics_json(evaluation["metrics"], metrics_json_path),
        "classification_report_csv": save_dataframe_csv(evaluation["classification_report_df"], classification_report_csv_path),
        "confusion_matrix_csv": save_dataframe_csv(evaluation["confusion_matrix_df"], confusion_matrix_csv_path)
    }

    return saved_paths


def main():
    y_true = [1, 2, 3, 1, 2, 3, 1]
    y_pred = [1, 2, 2, 1, 3, 3, 1]

    evaluation = evaluate_classification(y_true, y_pred)

    print("Basic metrics:")
    for k, v in evaluation["metrics"].items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()