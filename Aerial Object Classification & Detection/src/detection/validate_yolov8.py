from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO

from src.utils.io import save_json
from src.utils.logger import get_logger


def _safe_metric(metrics_obj, attr_name: str):
    attr = getattr(metrics_obj, attr_name, None)
    if attr is None:
        return None
    if callable(attr):
        try:
            return float(attr())
        except Exception:
            return None
    return attr


def validate_yolov8_detector(
    model_path: Path,
    data_yaml_path: Path,
    validation_config: Dict,
    logger=None,
) -> Dict:
    """
    Run YOLOv8 validation and collect core detection metrics.
    """
    logger = logger or get_logger(name="validate_yolov8")

    model_path = Path(model_path)
    data_yaml_path = Path(data_yaml_path)

    logger.info("Loading detector from %s", model_path)
    model = YOLO(str(model_path))

    logger.info("Running YOLOv8 validation on split=%s", validation_config["split"])
    metrics = model.val(
        data=str(data_yaml_path),
        split=validation_config.get("split", "test"),
        imgsz=validation_config.get("imgsz", 640),
        batch=validation_config.get("batch", 16),
        device=validation_config.get("device", 0),
        workers=validation_config.get("workers", 2),
        project=validation_config.get("project_dir", "models/detection"),
        name=validation_config.get("run_name", "yolov8_val"),
        exist_ok=True,
        plots=validation_config.get("plots", True),
        save_json=validation_config.get("save_json", True),
        verbose=True,
    )

    save_dir = Path(metrics.save_dir)

    payload = {
        "model_path": str(model_path),
        "data_yaml_path": str(data_yaml_path),
        "save_dir": str(save_dir),
        "metrics/mAP50": _safe_metric(metrics.box, "map50"),
        "metrics/mAP50-95": _safe_metric(metrics.box, "map"),
        "metrics/precision": _safe_metric(metrics.box, "mp"),
        "metrics/recall": _safe_metric(metrics.box, "mr"),
        "curves_results_dir": str(save_dir),
    }

    logger.info("Validation completed. Results at %s", save_dir)
    return payload


def save_validation_metrics(payload: Dict, output_path: Path) -> Path:
    """
    Save validation summary as JSON.
    """
    save_json(payload, output_path)
    return output_path