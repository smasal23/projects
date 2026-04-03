from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Tuple

from PIL import Image, UnidentifiedImageError


ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def validate_uploaded_image(uploaded_file, max_size_mb: int = 10) -> None:
    """
    Validate uploaded Streamlit image object.
    """
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    file_name = getattr(uploaded_file, "name", "uploaded_image")
    suffix = Path(file_name).suffix.lower()

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed types: {sorted(ALLOWED_IMAGE_SUFFIXES)}"
        )

    raw_bytes = uploaded_file.getvalue()
    max_bytes = max_size_mb * 1024 * 1024

    if len(raw_bytes) > max_bytes:
        raise ValueError(
            f"Uploaded file is too large ({len(raw_bytes) / (1024 * 1024):.2f} MB). "
            f"Maximum allowed size is {max_size_mb} MB."
        )

    try:
        image = Image.open(BytesIO(raw_bytes))
        image.verify()
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid readable image.") from exc
    except Exception as exc:
        raise ValueError(f"Uploaded image validation failed: {exc}") from exc


def load_pil_image_from_upload(uploaded_file) -> Image.Image:
    """
    Load uploaded file into RGB PIL image.
    """
    return Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")


def get_image_display_metadata(image: Image.Image) -> dict:
    """
    Extract display metadata from image.
    """
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4) if height else None,
        "mode": image.mode,
    }


def save_uploaded_image(
    uploaded_file,
    output_dir: Path,
    output_name: str | None = None,
) -> Path:
    """
    Persist uploaded image for detector/classifier workflows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_name = getattr(uploaded_file, "name", "uploaded_image.jpg")
    suffix = Path(original_name).suffix.lower() or ".jpg"
    file_name = output_name or Path(original_name).stem
    output_path = output_dir / f"{file_name}{suffix}"

    output_path.write_bytes(uploaded_file.getvalue())
    return output_path


def save_pil_image(image: Image.Image, output_path: Path) -> Path:
    """
    Save PIL image to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def resize_for_display(image: Image.Image, max_size: Tuple[int, int] = (900, 900)) -> Image.Image:
    """
    Resize image copy for UI display.
    """
    display_img = image.copy()
    display_img.thumbnail(max_size)
    return display_img