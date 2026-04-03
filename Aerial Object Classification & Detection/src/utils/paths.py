from pathlib import Path
from typing import Dict, Union

from src.utils.config import load_yaml


PathLike = Union[str, Path]


def get_project_root_from_drive(
    drive_root: str = "/content/drive/MyDrive",
    project_dir_name: str = "Aerial_Object_Classification_Detection",
) -> Path:
    """
    Build the project root path inside Google Drive.
    """
    return Path(drive_root) / project_dir_name


def build_project_paths(project_root: Path) -> Dict[str, Path]:
    """
    Build a dictionary of key project directories.
    """
    project_root = Path(project_root)

    return {
        "project_root": project_root,
        "data_root": project_root / "data",
        "raw_root": project_root / "data" / "raw",
        "interim_root": project_root / "data" / "interim",
        "processed_root": project_root / "data" / "processed",
        "classification_dataset": project_root / "data" / "raw" / "classification_dataset",
        "object_detection_dataset": project_root / "data" / "raw" / "object_detection_dataset",
        "processed_classification_root": project_root / "data" / "processed" / "classification",
        "processed_detection_root": project_root / "data" / "processed" / "detection",
        "dataset_audit_dir": project_root / "data" / "interim" / "dataset_audit",
        "previews_dir": project_root / "data" / "interim" / "previews",
        "label_checks_dir": project_root / "data" / "interim" / "label_checks",
        "notebooks_dir": project_root / "notebooks",
        "src_dir": project_root / "src",
        "configs_dir": project_root / "configs",
        "docs_dir": project_root / "docs",
        "reports_dir": project_root / "reports",
        "figures_dir": project_root / "figures",
        "figures_dataset_audit_dir": project_root / "figures" / "dataset_audit",
        "figures_preprocessing_dir": project_root / "figures" / "preprocessing",
        "app_dir": project_root / "app",
        "models_root": project_root / "models",
        "classification_models_dir": project_root / "models" / "classification",
        "detection_models_dir": project_root / "models" / "detection",
        "tests_dir": project_root / "tests",
        "logs_dir": project_root / "logs",
    }


def create_project_directories(paths: Dict[str, Path]) -> None:
    """
    Create all project directories if they do not exist.
    """
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def load_paths_config(config_path: PathLike) -> Dict[str, Path]:
    """
    Load paths.yaml and convert all configured filesystem paths to Path objects.
    """
    config = load_yaml(Path(config_path))
    path_dict = {}

    for section_name in ("paths", "files"):
        section = config.get(section_name, {})
        for key, value in section.items():
            path_dict[key] = Path(value)

    return path_dict


def resolve_project_path(path_value: PathLike) -> Path:
    """
    Convert a raw string or Path into Path.
    """
    return path_value if isinstance(path_value, Path) else Path(path_value)


def ensure_dir(path_value: PathLike) -> Path:
    """
    Ensure a directory exists and return it as Path.
    """
    path = resolve_project_path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path