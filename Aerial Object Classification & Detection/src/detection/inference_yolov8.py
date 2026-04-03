from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from ultralytics import YOLO

from src.utils.io import save_json, save_text
from src.utils.logger import get_logger


def run_yolov8_inference(
    model_path: Path,
    source: Path | str,
    inference_config: Dict,
    run_name: str,
    logger=None,
) -> Dict:
    """
    Run YOLOv8 inference on a file or directory source.
    """
    logger = logger or get_logger(name="inference_yolov8")

    model_path = Path(model_path)

    logger.info("Loading detector from %s", model_path)
    model = YOLO(str(model_path))

    logger.info("Running inference on source=%s", source)
    results = model.predict(
        source=str(source),
        conf=inference_config.get("conf", 0.25),
        save=inference_config.get("save", True),
        save_txt=inference_config.get("save_txt", False),
        line_width=inference_config.get("line_width", 2),
        project=inference_config.get("project_dir", "models/detection"),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(results[0].save_dir) if results else None

    payload = {
        "model_path": str(model_path),
        "source": str(source),
        "save_dir": None if save_dir is None else str(save_dir),
        "num_result_items": len(results),
    }

    return payload


def collect_prediction_image_paths(prediction_dir: Path) -> List[Path]:
    """
    Collect predicted JPG files from an inference output directory.
    """
    prediction_dir = Path(prediction_dir)
    if not prediction_dir.exists():
        return []
    return sorted(prediction_dir.glob("*.jpg"))


def build_inference_observations_note(
    test_payload: Dict,
    new_payload: Dict | None = None,
) -> str:
    """
    Build a short qualitative observations note for report usage.
    """
    lines = [
        "# YOLOv8 Inference Observations",
        "",
        "## Test-set inference",
        "",
        f"- Source: `{test_payload['source']}`",
        f"- Saved predictions directory: `{test_payload['save_dir']}`",
        f"- Number of result items: {test_payload['num_result_items']}",
        "",
        "The detector-generated images should be reviewed for box tightness, missed detections, false positives, and class consistency under scale and background variation.",
    ]

    if new_payload is not None:
        lines.extend([
            "",
            "## New-image inference",
            "",
            f"- Source: `{new_payload['source']}`",
            f"- Saved predictions directory: `{new_payload['save_dir']}`",
            f"- Number of result items: {new_payload['num_result_items']}",
            "",
            "For new images, focus on generalization quality: whether the detector remains stable on unseen framing, clutter, object size, and lighting conditions.",
        ])

    return "\n".join(lines)


def save_inference_summary(payload: Dict, output_path: Path) -> Path:
    save_json(payload, output_path)
    return output_path


def save_inference_observations(note: str, output_path: Path) -> Path:
    save_text(note, output_path)
    return output_path