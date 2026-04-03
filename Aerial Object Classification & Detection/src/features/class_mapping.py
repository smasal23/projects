from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.io import save_json


def build_class_to_index(class_names: List[str]) -> Dict[str, int]:
    """
    Build deterministic class-to-index mapping.
    """
    class_names = sorted(class_names)
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def build_index_to_class(class_to_index: Dict[str, int]) -> Dict[int, str]:
    """
    Reverse class mapping.
    """
    return {idx: class_name for class_name, idx in class_to_index.items()}


def build_class_mappings(class_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Return both forward and reverse mappings.
    """
    class_to_index = build_class_to_index(class_names)
    index_to_class = build_index_to_class(class_to_index)
    return class_to_index, index_to_class


def validate_expected_vs_found_classes(
    expected_classes: List[str],
    found_classes: List[str],
) -> None:
    """
    Ensure dataset classes match what config expects.
    """
    expected_sorted = sorted(expected_classes)
    found_sorted = sorted(found_classes)

    if expected_sorted != found_sorted:
        raise ValueError(
            f"Expected classes {expected_sorted}, but found {found_sorted}"
        )


def save_class_mapping_artifact(
    class_to_index: Dict[str, int],
    index_to_class: Dict[int, str],
    output_path: Path,
) -> None:
    """
    Save both mappings to JSON.
    """
    serializable = {
        "class_to_index": class_to_index,
        "index_to_class": {str(k): v for k, v in index_to_class.items()},
    }
    save_json(serializable, output_path)