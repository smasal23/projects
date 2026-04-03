from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

from src.features.augmentations import build_eval_preprocessing_pipeline
from src.features.preprocessing_transforms import build_tf_transfer_preprocess_layer
from src.utils.io import load_json


def load_class_mapping(mapping_path: Path) -> Dict:
    """
    Load class mapping JSON and normalize keys.
    """
    payload = load_json(Path(mapping_path))

    class_to_index = payload.get("class_to_index", {})
    raw_index_to_class = payload.get("index_to_class", {})
    index_to_class = {int(k): v for k, v in raw_index_to_class.items()}

    if not index_to_class and class_to_index:
        index_to_class = {int(v): k for k, v in class_to_index.items()}

    if not class_to_index and index_to_class:
        class_to_index = {v: int(k) for k, v in index_to_class.items()}

    class_names = [index_to_class[idx] for idx in sorted(index_to_class.keys())]

    return {
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
        "class_names": class_names,
    }


def build_classifier_registry(project_root: Path, streamlit_config: Dict) -> Dict[str, Dict]:
    """
    Build classifier registry from streamlit config only.
    """
    models_cfg = streamlit_config.get("classification", {}).get("models", {})
    registry = {}

    for model_key, model_cfg in models_cfg.items():
        if not model_cfg.get("enabled", True):
            continue

        registry[model_key] = {
            "display_name": model_cfg["display_name"],
            "model_path": project_root / model_cfg["model_path"],
            "class_mapping_path": project_root / model_cfg["class_mapping_path"],
            "preprocess_mode": model_cfg["preprocess_mode"],
            "backbone_name": model_cfg.get("backbone_name"),
            "enabled": model_cfg.get("enabled", True),
        }

    return registry


def validate_classifier_entry(entry: Dict) -> None:
    """
    Validate a registry entry and its artifact files.
    """
    required_keys = [
        "display_name",
        "model_path",
        "class_mapping_path",
        "preprocess_mode",
    ]
    missing = [key for key in required_keys if key not in entry]
    if missing:
        raise ValueError(f"Classifier entry missing keys: {missing}")

    model_path = Path(entry["model_path"])
    class_mapping_path = Path(entry["class_mapping_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {model_path}")

    if not class_mapping_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    if entry["preprocess_mode"] not in {"custom", "transfer"}:
        raise ValueError(f"Unsupported preprocess_mode: {entry['preprocess_mode']}")


def resolve_enabled_classifier_registry(project_root: Path, streamlit_config: Dict) -> Dict[str, Dict]:
    """
    Return validated enabled registry.
    """
    raw_registry = build_classifier_registry(project_root=project_root, streamlit_config=streamlit_config)

    enabled = {}
    for model_key, entry in raw_registry.items():
        if not entry.get("enabled", True):
            continue
        validate_classifier_entry(entry)
        enabled[model_key] = entry

    return enabled


def load_keras_model(model_path: Path) -> tf.keras.Model:
    """
    Load saved Keras model.
    """
    return tf.keras.models.load_model(Path(model_path), compile=False)


def build_preprocess_layer(preprocess_mode: str, backbone_name: Optional[str] = None):
    """
    Resolve preprocessing layer for inference.
    """
    if preprocess_mode == "custom":
        return build_eval_preprocessing_pipeline()

    if preprocess_mode == "transfer":
        if not backbone_name:
            raise ValueError("backbone_name is required for transfer preprocessing.")
        return build_tf_transfer_preprocess_layer(backbone_name)

    raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")


def prepare_pil_image_for_model(
    image: Image.Image,
    image_size: tuple[int, int] = (224, 224),
) -> tf.Tensor:
    """
    Convert PIL image to batched TensorFlow input tensor.
    """
    image = image.convert("RGB").resize(tuple(image_size))
    arr = tf.keras.utils.img_to_array(image)
    arr = tf.expand_dims(arr, axis=0)
    arr = tf.cast(arr, tf.float32)
    return arr


def predict_with_loaded_classifier(
    model: tf.keras.Model,
    image: Image.Image,
    class_mapping: Dict,
    preprocess_mode: str,
    backbone_name: Optional[str] = None,
    image_size: tuple[int, int] = (224, 224),
) -> Dict:
    """
    Run single-image classification inference.
    """
    preprocess_layer = build_preprocess_layer(
        preprocess_mode=preprocess_mode,
        backbone_name=backbone_name,
    )

    model_input = prepare_pil_image_for_model(image=image, image_size=image_size)
    model_input = preprocess_layer(model_input, training=False)

    raw_pred = np.array(model.predict(model_input, verbose=0))

    class_names = class_mapping["class_names"]
    index_to_class = class_mapping["index_to_class"]

    if raw_pred.shape[-1] == 1:
        positive_score = float(raw_pred.reshape(-1)[0])
        negative_score = 1.0 - positive_score

        probabilities = {
            index_to_class[0]: negative_score,
            index_to_class[1]: positive_score,
        }
        pred_index = int(positive_score >= 0.5)
        confidence = probabilities[index_to_class[pred_index]]
    else:
        prob_vec = raw_pred.reshape(-1).astype(float)
        pred_index = int(np.argmax(prob_vec))
        confidence = float(np.max(prob_vec))
        probabilities = {
            index_to_class[idx]: float(prob_vec[idx]) for idx in sorted(index_to_class.keys())
        }

    predicted_label = index_to_class[pred_index]

    sorted_probabilities = dict(
        sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "predicted_index": pred_index,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "probabilities": sorted_probabilities,
        "preprocess_mode": preprocess_mode,
        "backbone_name": backbone_name,
        "image_size": tuple(image_size),
        "class_names": class_names,
    }