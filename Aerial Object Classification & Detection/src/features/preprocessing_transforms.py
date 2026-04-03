from typing import Dict, Sequence

import tensorflow as tf
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_normalization() -> transforms.Normalize:
    """
    Standard ImageNet normalization for PyTorch transfer learning models.
    """
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_pytorch_transfer_transform(
    image_size: Sequence[int] = (224, 224),
) -> transforms.Compose:
    """
    Transfer-learning preprocessing branch for PyTorch:
    resize -> tensor -> ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        get_imagenet_normalization(),
    ])


def get_supported_torchvision_weights() -> Dict[str, object]:
    """
    Optional registry if you later want model-specific weights/transforms.
    """
    return {
        "resnet18": ResNet18_Weights.DEFAULT,
        "resnet50": ResNet50_Weights.DEFAULT,
        "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
        "mobilenet_v3_small": MobileNet_V3_Small_Weights.DEFAULT,
    }


def get_tf_backbone_preprocess(backbone_name: str):
    """
    Return the correct TensorFlow/Keras preprocess_input function
    for a supported transfer-learning backbone.
    """
    backbone_name = backbone_name.lower()

    if backbone_name in {"mobilenet", "mobilenetv2"}:
        return tf.keras.applications.mobilenet_v2.preprocess_input

    if backbone_name == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input

    if backbone_name == "efficientnetb0":
        return tf.keras.applications.efficientnet.preprocess_input

    raise ValueError(f"Unsupported TensorFlow backbone for preprocessing: {backbone_name}")


def build_tf_transfer_preprocess_layer(backbone_name: str) -> tf.keras.layers.Layer:
    """
    Wrap model-specific preprocess_input inside a Lambda layer so it can
    be used directly inside tf.data pipelines or Keras models.
    """
    preprocess_fn = get_tf_backbone_preprocess(backbone_name)
    return tf.keras.layers.Lambda(
        lambda x: preprocess_fn(tf.cast(x, tf.float32)),
        name=f"{backbone_name.lower()}_preprocess",
    )