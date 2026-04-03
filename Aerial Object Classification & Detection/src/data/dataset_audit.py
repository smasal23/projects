from pathlib import Path
from typing import Dict, List, Tuple
import random
import shutil

import pandas as pd
from PIL import Image, UnidentifiedImageError


def inspect_folder_structure(root: Path) -> Dict[str, List[str]]:
    """
    Return a dictionary describing immediate subdirectories and files.
    """
    structure = {}
    if not root.exists():
        return structure

    for item in sorted(root.iterdir()):
        if item.is_dir():
            structure[item.name] = sorted([child.name for child in item.iterdir()])
    return structure


def count_classification_images_by_split_and_class(
    dataset_root: Path,
    expected_splits: List[str],
    expected_classes: List[str],
    extension: str = ".jpg",
) -> pd.DataFrame:
    """
    Count images in classification dataset by split and class.
    """
    records = []

    for split in expected_splits:
        for class_name in expected_classes:
            class_dir = dataset_root / split / class_name
            count = 0
            if class_dir.exists():
                count = len([p for p in class_dir.glob(f"*{extension}") if p.is_file()])

            records.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "image_count": count,
                }
            )

    return pd.DataFrame(records)


def collect_non_jpg_files(dataset_root: Path) -> List[Path]:
    """
    Find files that are not .jpg under classification dataset.
    """
    non_jpg_files = []
    for p in dataset_root.rglob("*"):
        if p.is_file() and p.suffix.lower() != ".jpg":
            non_jpg_files.append(p)
    return sorted(non_jpg_files)


def get_sample_images(
    class_dir: Path,
    n_samples: int = 8,
    seed: int = 42,
) -> List[Path]:
    """
    Return a reproducible sample of .jpg images from a class directory.
    """
    image_paths = sorted([p for p in class_dir.glob("*.jpg") if p.is_file()])
    if not image_paths:
        return []

    random.seed(seed)
    n = min(n_samples, len(image_paths))
    return random.sample(image_paths, n)


def summarize_image_sizes(image_paths: List[Path]) -> pd.DataFrame:
    """
    Summarize image dimensions for a list of images.
    """
    records = []

    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        records.append(
            {
                "file_name": path.name,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4) if height != 0 else None,
            }
        )

    return pd.DataFrame(records)


def build_classification_audit_summary(
    counts_df: pd.DataFrame,
    non_jpg_files: List[Path],
    expected_classes: List[str],
) -> str:
    """
    Create a markdown summary for the classification audit.
    """
    found_classes = sorted(counts_df["class_name"].unique().tolist())
    total_images = int(counts_df["image_count"].sum())

    lines = []
    lines.append("# Classification Dataset Audit Summary")
    lines.append("")
    lines.append(f"- Expected classes: {expected_classes}")
    lines.append(f"- Found classes: {found_classes}")
    lines.append(f"- Total counted .jpg images: {total_images}")
    lines.append(f"- Non-.jpg files found: {len(non_jpg_files)}")
    lines.append("")
    lines.append("## Split-wise counts")
    lines.append("")
    lines.append(counts_df.to_markdown(index=False))

    if non_jpg_files:
        lines.append("")
        lines.append("## Non-.jpg files")
        lines.append("")
        for f in non_jpg_files[:50]:
            lines.append(f"- {f}")

    return "\n".join(lines)


def is_valid_jpg_image(image_path: Path) -> Tuple[bool, str]:
    """
    Validate that a file is a readable JPEG image.
    """
    if not image_path.exists():
        return False, "file_does_not_exist"

    if image_path.suffix.lower() != ".jpg":
        return False, "invalid_extension"

    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.convert("RGB")
    except UnidentifiedImageError:
        return False, "unidentified_or_corrupt_image"
    except Exception as exc:
        return False, f"image_validation_error: {exc}"

    return True, "ok"


def collect_valid_classification_images(
    dataset_root: Path,
    expected_splits: List[str],
    expected_classes: List[str],
    extension: str = ".jpg",
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Scan the classification dataset and return:
    1. valid image records for export
    2. validation report dataframe for all discovered candidate files
    """
    valid_records: List[Dict] = []
    validation_rows: List[Dict] = []

    for split in expected_splits:
        for class_name in expected_classes:
            class_dir = dataset_root / split / class_name

            if not class_dir.exists():
                validation_rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "file_path": str(class_dir),
                        "file_name": None,
                        "is_valid": False,
                        "reason": "missing_class_directory",
                    }
                )
                continue

            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue

                if path.suffix.lower() != extension.lower():
                    validation_rows.append(
                        {
                            "split": split,
                            "class_name": class_name,
                            "file_path": str(path),
                            "file_name": path.name,
                            "is_valid": False,
                            "reason": "invalid_extension",
                        }
                    )
                    continue

                is_valid, reason = is_valid_jpg_image(path)

                validation_rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "file_path": str(path),
                        "file_name": path.name,
                        "is_valid": is_valid,
                        "reason": reason,
                    }
                )

                if is_valid:
                    valid_records.append(
                        {
                            "split": split,
                            "class_name": class_name,
                            "source_path": path,
                            "file_name": path.name,
                        }
                    )

    validation_df = pd.DataFrame(validation_rows)
    return valid_records, validation_df


def export_audited_classification_dataset(
    source_root: Path,
    processed_root: Path,
    expected_splits: List[str],
    expected_classes: List[str],
    extension: str = ".jpg",
    copy_files: bool = True,
    clean_processed_dir: bool = True,
) -> Dict[str, object]:
    """
    Export only audited-valid classification images from source_root to processed_root.

    Output structure:
    processed_root/
      train/bird
      train/drone
      valid/bird
      valid/drone
      test/bird
      test/drone
    """
    source_root = Path(source_root)
    processed_root = Path(processed_root)

    valid_records, validation_df = collect_valid_classification_images(
        dataset_root=source_root,
        expected_splits=expected_splits,
        expected_classes=expected_classes,
        extension=extension,
    )

    if clean_processed_dir and processed_root.exists():
        shutil.rmtree(processed_root)

    for split in expected_splits:
        for class_name in expected_classes:
            (processed_root / split / class_name).mkdir(parents=True, exist_ok=True)

    copied_count = 0
    exported_rows: List[Dict] = []

    for row in valid_records:
        src_path = row["source_path"]
        dst_path = processed_root / row["split"] / row["class_name"] / row["file_name"]

        if copy_files:
            shutil.copy2(src_path, dst_path)
        else:
            dst_path.symlink_to(src_path)

        copied_count += 1
        exported_rows.append(
            {
                "split": row["split"],
                "class_name": row["class_name"],
                "source_path": str(src_path),
                "processed_path": str(dst_path),
                "file_name": row["file_name"],
            }
        )

    exported_df = pd.DataFrame(exported_rows)

    processed_counts_df = count_classification_images_by_split_and_class(
        dataset_root=processed_root,
        expected_splits=expected_splits,
        expected_classes=expected_classes,
        extension=extension,
    )

    invalid_df = validation_df.loc[validation_df["is_valid"] == False].copy()

    summary = {
        "source_root": source_root,
        "processed_root": processed_root,
        "num_valid_exported": copied_count,
        "num_invalid_or_skipped": int((~validation_df["is_valid"]).sum()) if not validation_df.empty else 0,
        "validation_df": validation_df,
        "invalid_df": invalid_df,
        "exported_df": exported_df,
        "processed_counts_df": processed_counts_df,
    }

    return summary


def build_processed_export_summary(
    source_root: Path,
    processed_root: Path,
    validation_df: pd.DataFrame,
    processed_counts_df: pd.DataFrame,
) -> str:
    """
    Create a markdown summary for the audited export step.
    """
    total_checked = len(validation_df)
    total_valid = int(validation_df["is_valid"].sum()) if not validation_df.empty else 0
    total_invalid = total_checked - total_valid

    lines = []
    lines.append("# Audited Classification Dataset Export Summary")
    lines.append("")
    lines.append(f"- Source root: `{source_root}`")
    lines.append(f"- Processed root: `{processed_root}`")
    lines.append(f"- Total files checked: {total_checked}")
    lines.append(f"- Valid exported files: {total_valid}")
    lines.append(f"- Invalid or skipped files: {total_invalid}")
    lines.append("")
    lines.append("## Processed split-wise counts")
    lines.append("")
    lines.append(processed_counts_df.to_markdown(index=False))

    invalid_df = validation_df.loc[validation_df["is_valid"] == False]
    if not invalid_df.empty:
        lines.append("")
        lines.append("## Invalid / skipped entries")
        lines.append("")
        preview_df = invalid_df[["split", "class_name", "file_name", "reason"]].head(100)
        lines.append(preview_df.to_markdown(index=False))

    return "\n".join(lines)