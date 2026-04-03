#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/content/drive/MyDrive/Aerial_Object_Classification_Detection}"

cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path

from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.modeling.train_classifier import train_custom_cnn_pipeline

project_root = Path(".").resolve()
logger = get_logger(name="run_custom_cnn")

classification_config = load_yaml(project_root / "configs" / "classification_config.yaml")
custom_cnn_config = classification_config["custom_cnn"]

processed_classification_root = project_root / "data" / "processed" / "classification"

set_seed(custom_cnn_config["training"]["seed"])

artifacts = train_custom_cnn_pipeline(
    dataset_root=processed_classification_root,
    custom_cnn_config=custom_cnn_config,
    project_root=project_root,
    logger=logger,
)

print("Training complete.")
print("Best model:", artifacts["best_model_path"])
print("Final model:", artifacts["final_model_path"])
print("History:", artifacts["history_path"])
print("Metrics:", artifacts["metrics_path"])
print("Report:", artifacts["report_path"])
PY