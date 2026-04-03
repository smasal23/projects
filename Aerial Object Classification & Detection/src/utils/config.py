from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return it as a dictionary.
    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary to a YAML file.
    """
    if not isinstance(path, Path):
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_project_configs(configs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Convenience loader for the configs used in this project.
    """
    if not isinstance(configs_dir, Path):
        configs_dir = Path(configs_dir)

    config_map = {
        "paths": configs_dir / "paths.yaml",
        "classification": configs_dir / "classification_config.yaml",
        "transfer_learning": configs_dir / "transfer_learning_config.yaml",
        "detection": configs_dir / "detection_config.yaml",
        "streamlit": configs_dir / "streamlit_config.yaml",
    }

    return {name: load_yaml(path) for name, path in config_map.items()}