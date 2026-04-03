from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image
from ultralytics import YOLO


def load_detector(model_path: Path) -> YOLO:
    """
    Load YOLO detector.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    return YOLO(str(model_path))


def run_detector_on_image(
    detector: YOLO,
    image_path: Path,
    conf: float = 0.25,
    line_width: int = 2,
) -> Dict:
    """
    Run YOLO detection on one image.
    """
    image_path = Path(image_path)

    results = detector.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        save_txt=False,
        line_width=line_width,
        verbose=False,
    )

    if not results:
        raise ValueError("YOLO did not return any prediction result.")

    result = results[0]
    annotated_bgr = result.plot(line_width=line_width)
    annotated_rgb = annotated_bgr[..., ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    detections: List[Dict] = []
    names_map = result.names if hasattr(result, "names") else {}

    if result.boxes is not None:
        xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        clss = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

        for box, score, cls_id in zip(xyxy, confs, clss):
            cls_id = int(cls_id)
            label = names_map.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]

            detections.append(
                {
                    "class_id": cls_id,
                    "label": label,
                    "confidence": float(score),
                    "xyxy": [x1, y1, x2, y2],
                }
            )

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    return {
        "num_detections": len(detections),
        "detections": detections,
        "annotated_image": annotated_image,
        "image_path": str(image_path),
        "confidence_threshold": conf,
    }