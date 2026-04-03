from pathlib import Path

from PIL import Image

from src.data.classification_loader import (
    build_classification_loader_pipeline,
    scan_classification_dataset,
)
from src.utils.config import load_yaml


def _create_dummy_image(path: Path, size=(300, 300), color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


def test_scan_classification_dataset_counts_images(tmp_path):
    dataset_root = tmp_path / "classification"

    for split in ["train", "valid", "test"]:
        for class_name in ["bird", "drone"]:
            _create_dummy_image(dataset_root / split / class_name / f"{class_name}_1.jpg")

    scan_df = scan_classification_dataset(
        dataset_root=dataset_root,
        expected_splits=["train", "valid", "test"],
        expected_classes=["bird", "drone"],
        allowed_extensions=[".jpg"],
    )

    assert scan_df["image_count"].sum() == 6
    assert scan_df["class_dir_exists"].all()


def test_build_classification_loader_pipeline(tmp_path):
    dataset_root = tmp_path / "classification"

    for split in ["train", "valid", "test"]:
        for class_name in ["bird", "drone"]:
            for idx in range(2):
                _create_dummy_image(
                    dataset_root / split / class_name / f"{class_name}_{idx}.jpg",
                    size=(280, 300),
                )

    classification_config = {
        "dataset": {
            "expected_splits": ["train", "valid", "test"],
            "expected_classes": ["bird", "drone"],
            "allowed_extensions": [".jpg"],
            "loader": {
                "image_size": [224, 224],
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_train": True,
                "drop_last_train": False,
            },
            "validation": {
                "require_non_empty_split": True,
                "require_all_expected_classes_per_split": True,
            },
        }
    }

    transfer_learning_config = {
        "training": {
            "image_size": [224, 224],
            "batch_size": 2,
        }
    }

    artifacts = build_classification_loader_pipeline(
        dataset_root=dataset_root,
        classification_config=classification_config,
        transfer_learning_config=transfer_learning_config,
    )

    train_images, train_labels = next(iter(artifacts["dataloaders"]["train"]))

    assert train_images.shape[1:] == (3, 224, 224)
    assert train_images.min().item() >= 0.0
    assert train_images.max().item() <= 1.0
    assert set(artifacts["class_to_index"].keys()) == {"bird", "drone"}
    assert set(artifacts["index_to_class"].values()) == {"bird", "drone"}