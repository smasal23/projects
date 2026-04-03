from __future__ import annotations

from pathlib import Path
from typing import Dict

from ultralytics import YOLO

from src.utils.io import save_json
from src.utils.logger import get_logger


def train_yolov8_detector(
    data_yaml_path: Path,
    training_config: Dict,
    logger=None,
) -> Dict:
    """
    Train a YOLOv8 detector using Ultralytics and return run metadata.
    """
    logger = logger or get_logger(name="train_yolov8")

    data_yaml_path = Path(data_yaml_path)
    model_size = training_config["model_size"]

    logger.info("Initializing YOLO model from %s", model_size)
    model = YOLO(model_size)

    logger.info("Starting YOLOv8 training...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=training_config["epochs"],
        imgsz=training_config["imgsz"],
        batch=training_config["batch"],
        patience=training_config["patience"],
        workers=training_config["workers"],
        device=training_config["device"],
        project=training_config["project_dir"],
        name=training_config["run_name"],
        exist_ok=training_config.get("exist_ok", True),
        verbose=training_config.get("verbose", True),
        seed=training_config.get("seed", 42),
        plots=True,
        save=True,
    )

    save_dir = Path(results.save_dir)

    payload = {
        "model_size": model_size,
        "data_yaml_path": str(data_yaml_path),
        "save_dir": str(save_dir),
        "best_checkpoint": str(save_dir / "weights" / "best.pt"),
        "last_checkpoint": str(save_dir / "weights" / "last.pt"),
        "results_csv": str(save_dir / "results.csv"),
        "args_yaml": str(save_dir / "args.yaml"),
    }

    logger.info("YOLOv8 training completed. Save dir: %s", save_dir)
    return payload


def save_training_run_metadata(payload: Dict, output_path: Path) -> Path:
    """
    Save run metadata for reproducibility.
    """
    save_json(payload, output_path)
    return output_path