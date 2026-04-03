from pathlib import Path
from typing import Dict, Sequence
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from src.features.augmentations import (
    build_augmentation_config_summary,
    build_eval_preprocessing_pipeline,
    build_train_augmentation_pipeline,
)
from src.features.class_mapping import build_class_mappings, save_class_mapping_artifact
from src.modeling.callbacks import build_training_callbacks
from src.modeling.custom_cnn import build_custom_cnn
from src.modeling.losses_metrics import (
    build_optimizer,
    evaluate_model_on_dataset,
    resolve_loss_and_metrics,
)
from src.utils.io import save_json, save_text
from src.utils.logger import get_logger


AUTOTUNE = tf.data.AUTOTUNE


def build_tf_directory_datasets(
    dataset_root: Path,
    image_size: Sequence[int] = (224, 224),
    batch_size: int = 32,
    label_mode: str = "binary",
    seed: int = 42,
):
    """
    Build raw TensorFlow datasets from train/valid/test directories.
    """
    dataset_root = Path(dataset_root)

    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    test_dir = dataset_root / "test"

    common_kwargs = {
        "labels": "inferred",
        "label_mode": label_mode,
        "image_size": tuple(image_size),
        "batch_size": batch_size,
    }

    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        seed=seed,
        **common_kwargs,
    )
    class_names = list(train_raw.class_names)

    valid_raw = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        shuffle=False,
        **common_kwargs,
    )
    test_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        **common_kwargs,
    )

    return {
        "train_raw": train_raw,
        "valid_raw": valid_raw,
        "test_raw": test_raw,
        "class_names": class_names,
    }


def apply_preprocessing_to_datasets(
    raw_datasets: Dict,
    train_preprocess: tf.keras.Model,
    eval_preprocess: tf.keras.Model,
) -> Dict:
    """
    Apply augmentation/preprocessing and prefetch datasets.
    """
    train_ds = raw_datasets["train_raw"].map(
        lambda x, y: (train_preprocess(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    valid_ds = raw_datasets["valid_raw"].map(
        lambda x, y: (eval_preprocess(x, training=False), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    test_ds = raw_datasets["test_raw"].map(
        lambda x, y: (eval_preprocess(x, training=False), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    return {
        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds,
        "class_names": raw_datasets["class_names"],
    }


def plot_training_history(history: Dict, accuracy_path: Path, loss_path: Path, dpi: int = 150) -> None:
    """
    Save training/validation accuracy and loss curves.
    """
    accuracy_path = Path(accuracy_path)
    loss_path = Path(loss_path)

    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    loss_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Custom CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title("Custom CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def infer_training_diagnostics(history: Dict) -> str:
    """
    Simple rule-based note for the report.
    """
    if not history.get("accuracy") or not history.get("val_accuracy"):
        return "Training diagnostics unavailable."

    train_acc_last = history["accuracy"][-1]
    val_acc_last = history["val_accuracy"][-1]
    train_loss_last = history["loss"][-1]
    val_loss_last = history["val_loss"][-1]

    acc_gap = train_acc_last - val_acc_last

    if acc_gap > 0.08 and train_loss_last < val_loss_last:
        return (
            "There are visible signs of overfitting: training accuracy is meaningfully above "
            "validation accuracy while validation loss remains higher than training loss."
        )

    if train_acc_last < 0.70 and val_acc_last < 0.70:
        return (
            "The model may still be underfitting: both training and validation accuracy remain modest, "
            "suggesting capacity, augmentation strength, or training duration may need adjustment."
        )

    return (
        "The training curves look reasonably stable without strong immediate evidence of severe "
        "overfitting or underfitting."
    )


def build_custom_cnn_report_text(
    config: Dict,
    class_names,
    history: Dict,
    validation_metrics: Dict,
    test_metrics: Dict,
    training_time_seconds: float | None = None,
) -> str:
    model_cfg = config["model"]
    aug_cfg = config["augmentation"]
    training_cfg = config["training"]

    report_lines = [
        "# Custom CNN Report",
        "",
        "## Experiment Summary",
        "",
        f"- Classes: {list(class_names)}",
        f"- Image size: {training_cfg['image_size']}",
        f"- Batch size: {training_cfg['batch_size']}",
        f"- Epochs configured: {training_cfg['epochs']}",
        f"- Optimizer: {training_cfg['optimizer']}",
        f"- Learning rate: {training_cfg['learning_rate']}",
    ]

    if training_time_seconds is not None:
        report_lines.append(f"- Training time (seconds): {training_time_seconds:.2f}")

    report_lines.extend([
        "",
        "## Architecture Choice",
        "",
        (
            f"A custom CNN was used with {len(model_cfg['conv_filters'])} convolution blocks, "
            f"filters {model_cfg['conv_filters']}, batch normalization after each convolutional stage, "
            f"max pooling for spatial reduction, block-wise dropout {model_cfg['block_dropout_rates']}, "
            f"and a dense head with {model_cfg['dense_units']} units and classifier dropout "
            f"{model_cfg['classifier_dropout']}."
        ),
        "",
        "## Augmentation Choice",
        "",
        (
            "The training pipeline applied "
            + build_augmentation_config_summary(aug_cfg)
            + ". These choices aim to improve robustness to pose variation, mirroring, "
              "small viewpoint changes, illumination changes, and mild spatial shifts."
        ),
        "",
        "## Early Observations",
        "",
        infer_training_diagnostics(history),
        "",
        "## Validation Metrics",
        "",
        f"- Accuracy: {validation_metrics['accuracy']:.4f}",
        f"- Precision: {validation_metrics['precision']:.4f}",
        f"- Recall: {validation_metrics['recall']:.4f}",
        f"- F1-score: {validation_metrics['f1_score']:.4f}",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Precision: {test_metrics['precision']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- F1-score: {test_metrics['f1_score']:.4f}",
    ])
    return "\n".join(report_lines)


def train_custom_cnn_pipeline(
    dataset_root: Path,
    custom_cnn_config: Dict,
    project_root: Path,
    logger=None,
) -> Dict:
    """
    End-to-end pipeline for:
    - TensorFlow dataset creation
    - augmentation setup
    - custom CNN build/compile
    - training
    - artifact saving
    - evaluation
    """
    logger = logger or get_logger(name="custom_cnn_training")

    dataset_root = Path(dataset_root)
    project_root = Path(project_root)

    training_cfg = custom_cnn_config["training"]
    aug_cfg = custom_cnn_config["augmentation"]
    model_cfg = custom_cnn_config["model"]
    callback_cfg = custom_cnn_config["callbacks"]
    artifact_cfg = custom_cnn_config["artifacts"]

    model_dir = project_root / artifact_cfg["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = project_root / artifact_cfg["model_dir"] / artifact_cfg["best_model_name"]
    final_model_path = project_root / artifact_cfg["model_dir"] / artifact_cfg["final_model_name"]
    history_path = project_root / artifact_cfg["model_dir"] / artifact_cfg["history_name"]
    metrics_path = project_root / artifact_cfg["model_dir"] / artifact_cfg["metrics_name"]
    report_path = project_root / artifact_cfg["report_path"]
    accuracy_curve_path = project_root / artifact_cfg["accuracy_curve_path"]
    loss_curve_path = project_root / artifact_cfg["loss_curve_path"]

    class_mapping_output_path = None
    if custom_cnn_config.get("dataset", {}).get("class_mapping", {}).get("save_mapping_json", False):
        class_mapping_output_path = model_dir / custom_cnn_config["dataset"]["class_mapping"].get(
            "mapping_file_name",
            "class_mapping.json",
        )

    logger.info("Building raw TensorFlow datasets...")
    raw_datasets = build_tf_directory_datasets(
        dataset_root=dataset_root,
        image_size=training_cfg["image_size"],
        batch_size=training_cfg["batch_size"],
        label_mode=training_cfg["label_mode"],
        seed=training_cfg["seed"],
    )

    logger.info("Building augmentation and preprocessing pipelines...")
    train_preprocess = build_train_augmentation_pipeline(
        image_size=training_cfg["image_size"],
        rotation_factor=aug_cfg["rotation_factor"],
        horizontal_flip=aug_cfg["horizontal_flip"],
        vertical_flip=aug_cfg["vertical_flip"],
        zoom_factor=aug_cfg["zoom_factor"],
        brightness_factor=aug_cfg["brightness_factor"],
        translation_height_factor=aug_cfg["translation_height_factor"],
        translation_width_factor=aug_cfg["translation_width_factor"],
    )
    eval_preprocess = build_eval_preprocessing_pipeline()

    logger.info("Applying preprocessing to datasets...")
    datasets = apply_preprocessing_to_datasets(
        raw_datasets=raw_datasets,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )

    logger.info("Building custom CNN model...")
    model = build_custom_cnn(
        input_shape=tuple(model_cfg["input_shape"]),
        num_classes=len(datasets["class_names"]),
        label_mode=training_cfg["label_mode"],
        conv_filters=model_cfg["conv_filters"],
        kernel_size=model_cfg["kernel_size"],
        dense_units=model_cfg["dense_units"],
        block_dropout_rates=model_cfg["block_dropout_rates"],
        classifier_dropout=model_cfg["classifier_dropout"],
        l2_regularization=model_cfg["l2_regularization"],
    )

    loss, metrics = resolve_loss_and_metrics(
        label_mode=training_cfg["label_mode"],
        num_classes=len(datasets["class_names"]),
        loss_name=training_cfg["loss"],
    )

    optimizer = build_optimizer(
        name=training_cfg["optimizer"],
        learning_rate=training_cfg["learning_rate"],
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    callbacks = build_training_callbacks(
        callback_config=callback_cfg,
        best_model_path=best_model_path,
    )

    logger.info("Starting model training...")
    start_time = time.perf_counter()

    history_obj = model.fit(
        datasets["train"],
        validation_data=datasets["valid"],
        epochs=training_cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    training_time_seconds = float(time.perf_counter() - start_time)

    history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
    save_json(history, history_path)

    logger.info("Saving final model...")
    model.save(final_model_path)

    logger.info("Evaluating validation dataset...")
    validation_metrics = evaluate_model_on_dataset(
        model=model,
        dataset=datasets["valid"],
        class_names=datasets["class_names"],
        label_mode=training_cfg["label_mode"],
    )

    logger.info("Evaluating test dataset...")
    test_metrics = evaluate_model_on_dataset(
        model=model,
        dataset=datasets["test"],
        class_names=datasets["class_names"],
        label_mode=training_cfg["label_mode"],
    )

    if class_mapping_output_path is not None:
        class_to_index, index_to_class = build_class_mappings(list(datasets["class_names"]))
        save_class_mapping_artifact(
            class_to_index=class_to_index,
            index_to_class=index_to_class,
            output_path=class_mapping_output_path,
        )

    best_epoch = int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i]) + 1)

    metrics_payload = {
        "model_name": "custom_cnn",
        "class_names": datasets["class_names"],
        "best_epoch": best_epoch,
        "training_time_seconds": training_time_seconds,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "class_mapping_path": None if class_mapping_output_path is None else str(class_mapping_output_path),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    save_json(metrics_payload, metrics_path)

    logger.info("Saving training curves...")
    plot_training_history(
        history=history,
        accuracy_path=accuracy_curve_path,
        loss_path=loss_curve_path,
    )

    logger.info("Writing report...")
    report_text = build_custom_cnn_report_text(
        config=custom_cnn_config,
        class_names=datasets["class_names"],
        history=history,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        training_time_seconds=training_time_seconds,
    )
    save_text(report_text, report_path)

    logger.info("Custom CNN pipeline completed successfully.")

    return {
        "model": model,
        "history": history,
        "class_names": datasets["class_names"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": training_time_seconds,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "class_mapping_path": class_mapping_output_path,
        "history_path": history_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "accuracy_curve_path": accuracy_curve_path,
        "loss_curve_path": loss_curve_path,
    }