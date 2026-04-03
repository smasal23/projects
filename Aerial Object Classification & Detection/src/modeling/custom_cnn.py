from typing import List, Sequence, Tuple

import tensorflow as tf


def resolve_output_layer(num_classes: int, label_mode: str = "binary") -> Tuple[int, str]:
    """
    Resolve output units and activation based on class setup.
    """
    if label_mode == "binary" or num_classes == 2:
        return 1, "sigmoid"
    return num_classes, "softmax"


def build_custom_cnn(
    input_shape: Sequence[int] = (224, 224, 3),
    num_classes: int = 2,
    label_mode: str = "binary",
    conv_filters: Sequence[int] = (32, 64, 128, 256),
    kernel_size: int = 3,
    dense_units: int = 128,
    block_dropout_rates: Sequence[float] = (0.10, 0.15, 0.20, 0.25),
    classifier_dropout: float = 0.40,
    l2_regularization: float = 1e-4,
) -> tf.keras.Model:
    """
    Build a custom CNN with:
    - input layer
    - 3 to 5 convolution blocks
    - max pooling
    - batch normalization
    - dropout
    - dense classifier head
    """
    if len(conv_filters) < 3 or len(conv_filters) > 5:
        raise ValueError("conv_filters must define between 3 and 5 convolution blocks.")

    if len(block_dropout_rates) != len(conv_filters):
        raise ValueError("block_dropout_rates length must match conv_filters length.")

    output_units, output_activation = resolve_output_layer(
        num_classes=num_classes,
        label_mode=label_mode,
    )

    regularizer = tf.keras.regularizers.l2(l2_regularization)

    inputs = tf.keras.layers.Input(shape=tuple(input_shape), name="input_image")
    x = inputs

    for idx, (filters, dropout_rate) in enumerate(zip(conv_filters, block_dropout_rates), start=1):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizer,
            name=f"conv_{idx}_a",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{idx}_a")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{idx}_a")(x)

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizer,
            name=f"conv_{idx}_b",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{idx}_b")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{idx}_b")(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"maxpool_{idx}")(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate, name=f"dropout_{idx}")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = tf.keras.layers.Dense(
        units=dense_units,
        activation=None,
        kernel_regularizer=regularizer,
        name="dense_features",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="dense_bn")(x)
    x = tf.keras.layers.Activation("relu", name="dense_relu")(x)
    x = tf.keras.layers.Dropout(rate=classifier_dropout, name="classifier_dropout")(x)

    outputs = tf.keras.layers.Dense(
        units=output_units,
        activation=output_activation,
        name="classifier_output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn_classifier")
    return model