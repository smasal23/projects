from src.modeling.custom_cnn import build_custom_cnn


def test_custom_cnn_binary_output_shape():
    model = build_custom_cnn(
        input_shape=(224, 224, 3),
        num_classes=2,
        label_mode="binary",
        conv_filters=(32, 64, 128, 256),
        block_dropout_rates=(0.10, 0.15, 0.20, 0.25),
    )

    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"


def test_custom_cnn_multiclass_output_shape():
    model = build_custom_cnn(
        input_shape=(224, 224, 3),
        num_classes=4,
        label_mode="categorical",
        conv_filters=(32, 64, 128),
        block_dropout_rates=(0.10, 0.15, 0.20),
    )

    assert model.output_shape == (None, 4)
    assert model.layers[-1].activation.__name__ == "softmax"