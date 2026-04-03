from pathlib import Path
import tempfile

from src.data.validate_detection_labels import (
    validate_yolo_line,
    validate_yolo_label_file,
    image_to_label_path,
    label_to_image_path,
)


def test_validate_yolo_line_valid():
    is_valid, message = validate_yolo_line(["0", "0.5", "0.5", "0.2", "0.3"])
    assert is_valid is True
    assert message == "OK"


def test_validate_yolo_line_invalid_length():
    is_valid, message = validate_yolo_line(["0", "0.5", "0.5"])
    assert is_valid is False
    assert "Expected exactly 5 values" in message


def test_validate_yolo_line_invalid_range():
    is_valid, message = validate_yolo_line(["0", "1.5", "0.5", "0.2", "0.3"])
    assert is_valid is False
    assert "must be in [0, 1]" in message


def test_validate_yolo_label_file_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        label_path = Path(tmpdir) / "sample.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        rows = validate_yolo_label_file(label_path)
        assert len(rows) == 1
        assert rows[0]["is_valid"] is True


def test_image_and_label_path_conversion():
    image_path = Path("/tmp/train/images/example.jpg")
    label_path = Path("/tmp/train/labels/example.txt")

    assert image_to_label_path(image_path) == label_path
    assert label_to_image_path(label_path) == image_path