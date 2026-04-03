from pathlib import Path

import pandas as pd

from src.evaluation.classification_report_utils import classification_report_dict_to_dataframe
from src.evaluation.compare_models import build_model_comparison_table
from src.evaluation.error_analysis import build_prediction_analysis_table, split_binary_error_types


def test_classification_report_dict_to_dataframe():
    report_dict = {
        "bird": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 10},
        "drone": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 12},
        "accuracy": 0.85,
        "macro avg": {"precision": 0.85, "recall": 0.85, "f1-score": 0.85, "support": 22},
    }

    df = classification_report_dict_to_dataframe(report_dict)
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    assert "precision" in df.columns


def test_build_model_comparison_table_sorts_best_first():
    results = [
        {
            "model_name": "resnet50",
            "accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.89,
            "f1_score": 0.88,
            "generalization_gap_abs": 0.03,
            "speed_bucket": "slower",
        },
        {
            "model_name": "mobilenet",
            "accuracy": 0.91,
            "precision": 0.90,
            "recall": 0.92,
            "f1_score": 0.91,
            "generalization_gap_abs": 0.01,
            "speed_bucket": "fast",
        },
    ]

    df = build_model_comparison_table(results)
    assert df.iloc[0]["model_name"] == "mobilenet"
    assert "streamlit_suitability" in df.columns
    assert "selection_score" in df.columns


def test_error_analysis_binary_split():
    image_paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg"), Path("d.jpg")]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_score = [0.1, 0.8, 0.9, 0.95]
    index_to_class = {0: "bird", 1: "drone"}

    df = build_prediction_analysis_table(
        image_paths=image_paths,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        index_to_class=index_to_class,
    )

    tables = split_binary_error_types(
        prediction_df=df,
        negative_class_name="bird",
        positive_class_name="drone",
    )

    assert len(tables["false_positives"]) == 1
    assert len(tables["false_negatives"]) == 1
    assert len(tables["hard_images"]) == 2