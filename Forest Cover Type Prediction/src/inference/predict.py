"""
predict.py

# Goal
Load model artifact and run predictions for given input.

# Inputs
- models/best_model.joblib
- user input data (dict/json/df)

# Outputs
- predicted class
- confidence score (optional)

# Steps
1) Load model artifact.
2) Convert input to DataFrame.
3) Validate using schema.validate_input.
4) Predict class.
5) Predict proba (if available).
6) Return prediction + confidence.
7) Optional CLI entrypoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.inference.schema import validate_raw_input
from src.inference.feature import build_inference_features


default_model_path = Path("models/random_forest_final_model.joblib")

class PredictionError(RuntimeError):
    """Raised when prediction cannot be completed."""

def load_artifact(model_path: str | Path = default_model_path) -> Any:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def predict_records(
    data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model_path: str | Path = default_model_path,
):
    model = load_artifact(model_path)
    raw_df = validate_raw_input(data)
    feature_df = build_inference_features(raw_df)

    predictions = model.predict(feature_df)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_df)

    results: list[dict[str, Any]] = []

    for i, pred in enumerate(predictions):
        row_result: dict[str, Any] = {
            "row_index": int(raw_df.index[i]),
            "predicted_class": str(pred),
        }

        if probabilities is not None:
            max_prob = float(probabilities[i].max())
            row_result["confidence"] = round(max_prob, 6)

        results.append(row_result)

    return results


def load_json_input(json_path: str | Path) -> list[dict[str, Any]] | dict[str, Any]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run inference using a saved EcoType model.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file containing one record or a list of records.",
    )
    parser.add_argument(
        "--model",
        default=str(default_model_path),
        help="Path to saved model artifact (.joblib).",
    )

    args = parser.parse_args()

    input_data = load_json_input(args.input)
    results = predict_records(data=input_data, model_path=args.model)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
