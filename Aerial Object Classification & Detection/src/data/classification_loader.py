from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.features.class_mapping import (
    build_class_mappings,
    validate_expected_vs_found_classes,
)
from src.features.preprocessing_transforms import build_pytorch_transfer_transform
from src.utils.helpers import is_jpg_file, sorted_subdirs
from src.utils.io import list_files_recursive
from src.utils.logger import get_logger


class ClassificationImageDataset(Dataset):
    """
    Custom PyTorch dataset for directory-structured classification data.

    Expected layout:
    root/
      train/
        bird/
        drone/
      valid/
        bird/
        drone/
      test/
        bird/
        drone/
    """

    def __init__(
        self,
        split_dir: Path,
        class_to_index: Dict[str, int],
        transform=None,
        allowed_extensions: Optional[Sequence[str]] = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.class_to_index = class_to_index
        self.transform = transform
        self.allowed_extensions = {ext.lower() for ext in (allowed_extensions or [".jpg"])}

        self.samples = self._scan_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No image samples found in split directory: {self.split_dir}")

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        samples = []

        for class_name, class_index in sorted(self.class_to_index.items(), key=lambda x: x[1]):
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue

            image_paths = [
                path for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in self.allowed_extensions
            ]

            for image_path in image_paths:
                samples.append((image_path, class_index))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    @property
    def class_names(self) -> List[str]:
        return [class_name for class_name, _ in sorted(self.class_to_index.items(), key=lambda x: x[1])]


def scan_classification_dataset(
    dataset_root: Path,
    expected_splits: Sequence[str],
    expected_classes: Sequence[str],
    allowed_extensions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Scan dataset and return image counts by split and class.
    """
    dataset_root = Path(dataset_root)
    allowed_extensions = {ext.lower() for ext in (allowed_extensions or [".jpg"])}

    records = []

    for split in expected_splits:
        split_dir = dataset_root / split
        for class_name in expected_classes:
            class_dir = split_dir / class_name
            image_count = 0

            if class_dir.exists():
                image_count = len([
                    p for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in allowed_extensions
                ])

            records.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "image_count": image_count,
                    "class_dir_exists": class_dir.exists(),
                }
            )

    return pd.DataFrame(records)


def count_images_in_split(split_dir: Path, allowed_extensions: Optional[Sequence[str]] = None) -> int:
    """
    Count images recursively within a split directory.
    """
    allowed_extensions = list(allowed_extensions or [".jpg"])
    return len(list_files_recursive(Path(split_dir), suffixes=allowed_extensions))


def discover_classes_from_train_split(train_dir: Path) -> List[str]:
    """
    Discover class names from immediate subdirectories of the train split.
    """
    return [p.name for p in sorted_subdirs(Path(train_dir))]


def validate_classification_dataset_structure(
    dataset_root: Path,
    expected_splits: Sequence[str],
    expected_classes: Sequence[str],
    require_non_empty_split: bool = True,
    require_all_expected_classes_per_split: bool = True,
    allowed_extensions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Validate split/class folder layout and image presence.
    """
    scan_df = scan_classification_dataset(
        dataset_root=dataset_root,
        expected_splits=expected_splits,
        expected_classes=expected_classes,
        allowed_extensions=allowed_extensions,
    )

    if require_all_expected_classes_per_split:
        missing_class_dirs = scan_df.loc[~scan_df["class_dir_exists"]]
        if not missing_class_dirs.empty:
            missing_rows = missing_class_dirs[["split", "class_name"]].to_dict(orient="records")
            raise ValueError(f"Missing class folders detected: {missing_rows}")

    if require_non_empty_split:
        split_totals = scan_df.groupby("split", as_index=False)["image_count"].sum()
        empty_splits = split_totals.loc[split_totals["image_count"] == 0, "split"].tolist()
        if empty_splits:
            raise ValueError(f"Empty dataset splits detected: {empty_splits}")

    return scan_df


def build_pytorch_classification_datasets(
    dataset_root: Path,
    expected_splits: Sequence[str],
    expected_classes: Sequence[str],
    image_size: Sequence[int] = (224, 224),
    allowed_extensions: Optional[Sequence[str]] = None,
    use_transfer_preprocessing: bool = False,
):
    """
    Build train/valid/test PyTorch datasets.
    """
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "train"

    found_classes = discover_classes_from_train_split(train_dir)
    validate_expected_vs_found_classes(
        expected_classes=list(expected_classes),
        found_classes=found_classes,
    )

    class_to_index, index_to_class = build_class_mappings(found_classes)

    if use_transfer_preprocessing:
        transform = build_pytorch_transfer_transform(image_size=image_size)
    else:
        from src.data.preprocess_classification import build_basic_classification_transform
        transform = build_basic_classification_transform(image_size=image_size)

    datasets = {}
    for split in expected_splits:
        datasets[split] = ClassificationImageDataset(
            split_dir=dataset_root / split,
            class_to_index=class_to_index,
            transform=transform,
            allowed_extensions=allowed_extensions,
        )

    metadata = {
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
        "class_names": found_classes,
    }

    return datasets, metadata


def build_pytorch_classification_dataloaders(
    datasets: Dict[str, Dataset],
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    drop_last_train: bool = False,
) -> Dict[str, DataLoader]:
    """
    Build train/valid/test dataloaders from prepared datasets.
    """
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_train,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }
    return dataloaders


def build_classification_loader_pipeline(
    dataset_root: Path,
    classification_config: Dict,
    transfer_learning_config: Dict,
    logger=None,
):
    """
    End-to-end pipeline:
    1. Validate dataset structure
    2. Build datasets
    3. Build dataloaders
    4. Return mappings + scan summary
    """
    logger = logger or get_logger(name="classification_loader")

    expected_splits = classification_config["dataset"]["expected_splits"]
    expected_classes = classification_config["dataset"]["expected_classes"]
    allowed_extensions = classification_config["dataset"].get("allowed_extensions", [".jpg"])

    image_size = classification_config["dataset"]["loader"].get(
        "image_size",
        transfer_learning_config["training"]["image_size"],
    )
    batch_size = classification_config["dataset"]["loader"].get(
        "batch_size",
        transfer_learning_config["training"]["batch_size"],
    )
    num_workers = classification_config["dataset"]["loader"].get("num_workers", 2)
    pin_memory = classification_config["dataset"]["loader"].get("pin_memory", True)
    shuffle_train = classification_config["dataset"]["loader"].get("shuffle_train", True)
    drop_last_train = classification_config["dataset"]["loader"].get("drop_last_train", False)

    logger.info("Validating classification dataset structure...")
    scan_df = validate_classification_dataset_structure(
        dataset_root=dataset_root,
        expected_splits=expected_splits,
        expected_classes=expected_classes,
        require_non_empty_split=classification_config["dataset"]["validation"].get("require_non_empty_split", True),
        require_all_expected_classes_per_split=classification_config["dataset"]["validation"].get(
            "require_all_expected_classes_per_split", True
        ),
        allowed_extensions=allowed_extensions,
    )

    logger.info("Building PyTorch classification datasets...")
    datasets, metadata = build_pytorch_classification_datasets(
        dataset_root=dataset_root,
        expected_splits=expected_splits,
        expected_classes=expected_classes,
        image_size=image_size,
        allowed_extensions=allowed_extensions,
        use_transfer_preprocessing=False,
    )

    logger.info("Building PyTorch dataloaders...")
    dataloaders = build_pytorch_classification_dataloaders(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_train=shuffle_train,
        drop_last_train=drop_last_train,
    )

    logger.info("Classification loader pipeline built successfully.")

    return {
        "datasets": datasets,
        "dataloaders": dataloaders,
        "scan_df": scan_df,
        "class_to_index": metadata["class_to_index"],
        "index_to_class": metadata["index_to_class"],
        "class_names": metadata["class_names"],
        "image_size": tuple(image_size),
        "batch_size": batch_size,
    }