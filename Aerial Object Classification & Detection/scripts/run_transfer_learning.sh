#!/usr/bin/env bash
set -e

PROJECT_ROOT="/content/drive/MyDrive/Aerial_Object_Classification_Detection"

cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
import sys

PROJECT_ROOT = Path("/content/drive/MyDrive/Aerial_Object_Classification_Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.paths import load_paths_config
from src.modeling.transfer_learning import train_transfer_learning_suite

logger = get_logger(name="run_transfer_learning_script")

paths_config = load_paths_config(PROJECT_ROOT / "configs" / "paths.yaml")
transfer_learning_config = load_yaml(PROJECT_ROOT / "configs" / "transfer_learning_config.yaml")

processed_classification_root = paths_config.get(
    "processed_classification_root",
    paths_config["processed_root"] / "classification",
)

set_seed(transfer_learning_config["training"]["seed"])

artifacts = train_transfer_learning_suite(
    dataset_root=processed_classification_root,
    project_root=PROJECT_ROOT,
    transfer_learning_config=transfer_learning_config,
    logger=logger,
)

print("Saved report:", artifacts["report_path"])
print("Saved comparison table:", artifacts["comparison_table_path"])
print(artifacts["summary_df"])
PY