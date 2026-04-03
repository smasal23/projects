from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def save_current_figure(output_path: Path, dpi: int = 150) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_image_grid(
    image_paths: List[Path],
    title: str,
    output_path: Optional[Path] = None,
    ncols: int = 4,
    figsize: tuple = (16, 10),
    dpi: int = 150,
) -> None:
    """
    Plot a grid of images from file paths.
    """
    if not image_paths:
        raise ValueError("No images provided to plot_image_grid.")

    n_images = len(image_paths)
    nrows = math.ceil(n_images / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, image_path in zip(axes, image_paths):
        img = Image.open(image_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(image_path.name, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=16)

    if output_path:
        save_current_figure(output_path, dpi=dpi)
    else:
        plt.show()


def tensor_to_display_image(
    tensor: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Convert a CHW tensor to a displayable HWC numpy image.
    Optionally denormalize using mean/std.
    """
    if tensor.ndim != 3:
        raise ValueError("Expected a 3D tensor in CHW format.")

    image = tensor.detach().cpu().float().clone()

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
        image = image * std_t + mean_t

    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return image


def save_tensor_preview(
    tensor: torch.Tensor,
    output_path: Path,
    title: str,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    dpi: int = 150,
    figsize: Tuple[int, int] = (6, 6),
) -> None:
    """
    Save a single tensor image preview.
    """
    image = tensor_to_display_image(tensor=tensor, mean=mean, std=std)

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    save_current_figure(output_path=output_path, dpi=dpi)