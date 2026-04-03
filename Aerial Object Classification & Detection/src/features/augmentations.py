from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def build_train_augmentation_pipeline(
    image_size: Sequence[int] = (224, 224),
    rotation_factor: float = 0.08,
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    zoom_factor: float = 0.10,
    brightness_factor: float = 0.15,
    translation_height_factor: float = 0.08,
    translation_width_factor: float = 0.08,
) -> tf.keras.Sequential:
    """
    Build training-time augmentation pipeline.

    Notes:
    - Images from image_dataset_from_directory are already resized to image_size.
    - This pipeline rescales to [0,1] and applies stochastic augmentation.
    """
    flip_mode = None
    if horizontal_flip and vertical_flip:
        flip_mode = "horizontal_and_vertical"
    elif horizontal_flip:
        flip_mode = "horizontal"
    elif vertical_flip:
        flip_mode = "vertical"

    layers = [
        tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_to_unit_interval"),
    ]

    if rotation_factor > 0:
        layers.append(
            tf.keras.layers.RandomRotation(
                factor=rotation_factor,
                fill_mode="nearest",
                name="random_rotation",
            )
        )

    if flip_mode is not None:
        layers.append(
            tf.keras.layers.RandomFlip(
                mode=flip_mode,
                name="random_flip",
            )
        )

    if zoom_factor > 0:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=(-zoom_factor, zoom_factor),
                width_factor=(-zoom_factor, zoom_factor),
                fill_mode="nearest",
                name="random_zoom",
            )
        )

    if translation_height_factor > 0 or translation_width_factor > 0:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=translation_height_factor,
                width_factor=translation_width_factor,
                fill_mode="nearest",
                name="random_translation",
            )
        )

    if brightness_factor > 0:
        layers.append(
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(
                    tf.image.random_brightness(x, max_delta=brightness_factor),
                    0.0,
                    1.0,
                ),
                name="random_brightness",
            )
        )

    return tf.keras.Sequential(layers, name="train_augmentation_pipeline")


def build_eval_preprocessing_pipeline() -> tf.keras.Sequential:
    """
    Deterministic preprocessing for validation/test data.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_to_unit_interval"),
        ],
        name="eval_preprocessing_pipeline",
    )


def _to_numpy_image(image_tensor: tf.Tensor) -> np.ndarray:
    image = image_tensor.numpy()
    image = np.clip(image, 0.0, 1.0)
    return image


def save_augmentation_preview(
    image_paths,
    augmentation_model: tf.keras.Model,
    output_path: Path,
    image_size: Tuple[int, int] = (224, 224),
    n_samples: int = 4,
    dpi: int = 150,
) -> None:
    """
    Save a preview figure showing original images vs augmented outputs.
    """
    image_paths = list(image_paths)[:n_samples]
    if len(image_paths) == 0:
        raise ValueError("No image paths provided for augmentation preview.")

    originals = []
    augmented = []

    for image_path in image_paths:
        image = tf.keras.utils.load_img(image_path, target_size=tuple(image_size))
        image = tf.keras.utils.img_to_array(image)   # returns numpy array
        image_batch = tf.expand_dims(image, axis=0)

        original = image / 255.0
        aug_image = augmentation_model(image_batch, training=True)[0]

        originals.append(original)   # fixed
        augmented.append(_to_numpy_image(aug_image))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(image_paths),
        ncols=2,
        figsize=(10, 4 * len(image_paths)),
    )

    if len(image_paths) == 1:
        axes = np.array([axes])

    for row_idx, image_path in enumerate(image_paths):
        axes[row_idx, 0].imshow(originals[row_idx])
        axes[row_idx, 0].set_title(f"Original: {Path(image_path).name}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(augmented[row_idx])
        axes[row_idx, 1].set_title(f"Augmented: {Path(image_path).name}")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_augmentation_config_summary(config: Dict) -> str:
    """
    Build a compact text summary for notebook/report usage.
    """
    return (
        f"rotation={config['rotation_factor']}, "
        f"horizontal_flip={config['horizontal_flip']}, "
        f"vertical_flip={config['vertical_flip']}, "
        f"zoom_factor={config['zoom_factor']}, "
        f"brightness_factor={config['brightness_factor']}, "
        f"translation_height_factor={config['translation_height_factor']}, "
        f"translation_width_factor={config['translation_width_factor']}"
    )