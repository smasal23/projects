from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.utils.config import save_yaml
from src.utils.io import save_text


def build_yolo_data_yaml_dict(
    dataset_root: Path,
    class_names: List[str],
    include_test: bool = True,
) -> Dict:
    """
    Build a YOLO-compatible data.yaml dictionary.
    """
    dataset_root = Path(dataset_root)

    payload = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "names": {idx: name for idx, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    if include_test:
        payload["test"] = "test/images"

    return payload


def save_yolo_data_yaml(
    dataset_root: Path,
    class_names: List[str],
    output_path: Path,
    include_test: bool = True,
) -> Path:
    """
    Save YOLO data.yaml to disk.
    """
    payload = build_yolo_data_yaml_dict(
        dataset_root=dataset_root,
        class_names=class_names,
        include_test=include_test,
    )
    save_yaml(payload, output_path)
    return output_path


def build_data_yaml_markdown_preview(data_yaml_dict: Dict) -> str:
    """
    Build a markdown preview of the generated YOLO data.yaml content.
    """
    lines = [
        "# YOLO data.yaml Preview",
        "",
        f"- path: `{data_yaml_dict['path']}`",
        f"- train: `{data_yaml_dict['train']}`",
        f"- val: `{data_yaml_dict['val']}`",
        f"- nc: {data_yaml_dict['nc']}",
        f"- names: {data_yaml_dict['names']}",
    ]

    if "test" in data_yaml_dict:
        lines.append(f"- test: `{data_yaml_dict['test']}`")

    return "\n".join(lines)


def save_data_yaml_preview_markdown(data_yaml_dict: Dict, output_path: Path) -> Path:
    """
    Save a small markdown summary for report/debug usage.
    """
    save_text(build_data_yaml_markdown_preview(data_yaml_dict), output_path)
    return output_path