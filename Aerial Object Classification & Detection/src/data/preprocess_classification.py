from typing import Callable, Optional, Sequence, Tuple

from torchvision import transforms


def build_basic_classification_transform(
    image_size: Sequence[int] = (224, 224),
    normalize_to_unit_interval: bool = True,
) -> transforms.Compose:
    """
    Standard preprocessing:
    - Resize to target image size
    - Convert PIL image to tensor
    - Tensor values become [0, 1] automatically through ToTensor()
    """
    transform_steps = [
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
    ]

    if not normalize_to_unit_interval:
        # Kept for API symmetry, though ToTensor() already scales to [0,1].
        pass

    return transforms.Compose(transform_steps)


def build_resize_only_transform(
    image_size: Sequence[int] = (224, 224),
) -> transforms.Compose:
    """
    Resize but keep PIL output for preview/debug workflows.
    """
    return transforms.Compose([
        transforms.Resize(tuple(image_size)),
    ])


def build_standard_train_transform(
    image_size: Sequence[int] = (224, 224),
) -> transforms.Compose:
    """
    Training transform for baseline preprocessing in this phase.
    No augmentation yet, only deterministic resizing and tensor conversion.
    """
    return build_basic_classification_transform(image_size=image_size)


def build_standard_eval_transform(
    image_size: Sequence[int] = (224, 224),
) -> transforms.Compose:
    """
    Validation/test transform for baseline preprocessing in this phase.
    """
    return build_basic_classification_transform(image_size=image_size)