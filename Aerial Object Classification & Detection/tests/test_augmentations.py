import tensorflow as tf

from src.features.augmentations import (
    build_eval_preprocessing_pipeline,
    build_train_augmentation_pipeline,
)


def test_train_augmentation_pipeline_preserves_shape():
    x = tf.random.uniform(shape=(4, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)

    augmenter = build_train_augmentation_pipeline(
        image_size=(224, 224),
        rotation_factor=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_factor=0.1,
        brightness_factor=0.1,
        translation_height_factor=0.05,
        translation_width_factor=0.05,
    )

    y = augmenter(x, training=True)

    assert y.shape == x.shape
    assert float(tf.reduce_min(y).numpy()) >= 0.0
    assert float(tf.reduce_max(y).numpy()) <= 1.0


def test_eval_preprocessing_pipeline_scales_to_unit_interval():
    x = tf.ones(shape=(2, 224, 224, 3), dtype=tf.float32) * 255.0

    preprocess = build_eval_preprocessing_pipeline()
    y = preprocess(x, training=False)

    assert y.shape == x.shape
    assert float(tf.reduce_min(y).numpy()) >= 0.0
    assert float(tf.reduce_max(y).numpy()) <= 1.0