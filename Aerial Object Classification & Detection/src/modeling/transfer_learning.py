from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.features.preprocessing_transforms import build_tf_transfer_preprocess_layer
from src.modeling.callbacks import build_training_callbacks
from src.modeling.losses_metrics import (
    build_optimizer,
    evaluate_model_on_dataset,
    resolve_loss_and_metrics,
)
from src.modeling.train_classifier import build_tf_directory_datasets
from src.utils.io import save_dataframe, save_json, save_text
from src.utils.logger import get_logger


AUTOTUNE = tf.data.AUTOTUNE


def get_supported_tf_backbone(backbone_name: str):
    """
    Return a TensorFlow/Keras application constructor for a supported backbone.
    """
    backbone_name = backbone_name.lower()

    if backbone_name in {"mobilenet", "mobilenetv2"}:
        return tf.keras.applications.MobileNetV2
    if backbone_name == "resnet50":
        return tf.keras.applications.ResNet50
    if backbone_name == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0

    raise ValueError(f"Unsupported transfer-learning backbone: {backbone_name}")


def build_transfer_datasets(
    dataset_root: Path,
    image_size: Sequence[int],
    batch_size: int,
    label_mode: str,
    seed: int,
    backbone_name: str,
) -> Dict:
    """
    Build TensorFlow directory datasets and apply backbone-specific preprocessing.
    """
    raw_datasets = build_tf_directory_datasets(
        dataset_root=dataset_root,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
        seed=seed,
    )

    preprocess_layer = build_tf_transfer_preprocess_layer(backbone_name)

    train_ds = raw_datasets["train_raw"].map(
        lambda x, y: (preprocess_layer(x, training=False), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    valid_ds = raw_datasets["valid_raw"].map(
        lambda x, y: (preprocess_layer(x, training=False), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    test_ds = raw_datasets["test_raw"].map(
        lambda x, y: (preprocess_layer(x, training=False), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    return {
        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds,
        "class_names": raw_datasets["class_names"],
    }


def resolve_output_layer(num_classes: int, label_mode: str = "binary") -> Tuple[int, str]:
    """
    Resolve output units and activation based on class setup.
    """
    if label_mode == "binary" or num_classes == 2:
        return 1, "sigmoid"
    return num_classes, "softmax"


def build_classification_head(
    x: tf.Tensor,
    dense_units: int,
    dropout_rate: float,
    l2_regularization: float,
    num_classes: int,
    label_mode: str,
) -> tf.Tensor:
    """
    Build a standard dense classification head on top of the backbone output.
    """
    output_units, output_activation = resolve_output_layer(
        num_classes=num_classes,
        label_mode=label_mode,
    )

    regularizer = tf.keras.regularizers.l2(l2_regularization)

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name="transfer_dense",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="transfer_dense_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="transfer_dropout")(x)

    outputs = tf.keras.layers.Dense(
        output_units,
        activation=output_activation,
        name="transfer_classifier_output",
    )(x)

    return outputs


def build_transfer_learning_model(
    backbone_name: str,
    input_shape: Sequence[int],
    weights: str,
    include_top: bool,
    pooling: str,
    dense_units: int,
    dropout_rate: float,
    l2_regularization: float,
    num_classes: int,
    label_mode: str,
    freeze_backbone: bool = True,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build a transfer-learning classifier and return:
    1. full model
    2. base backbone model
    """
    backbone_constructor = get_supported_tf_backbone(backbone_name)

    inputs = tf.keras.layers.Input(shape=tuple(input_shape), name="input_image")

    base_model = backbone_constructor(
        include_top=include_top,
        weights=weights,
        input_shape=tuple(input_shape),
        pooling=pooling,
    )
    base_model.trainable = not freeze_backbone

    x = base_model(inputs, training=False)
    outputs = build_classification_head(
        x=x,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
        num_classes=num_classes,
        label_mode=label_mode,
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{backbone_name.lower()}_transfer_classifier",
    )
    return model, base_model


def compile_transfer_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer_name: str,
    label_mode: str,
    num_classes: int,
) -> None:
    """
    Compile the transfer-learning model using shared optimizer/loss/metrics utilities.
    """
    loss, metrics = resolve_loss_and_metrics(
        label_mode=label_mode,
        num_classes=num_classes,
        loss_name="auto",
    )

    optimizer = build_optimizer(
        name=optimizer_name,
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )


def unfreeze_top_layers(base_model: tf.keras.Model, n_layers: int) -> None:
    """
    Unfreeze only the top N layers of the backbone while keeping BatchNorm layers frozen.
    """
    if n_layers <= 0:
        return

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-n_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def merge_history_dicts(history_parts: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    """
    Merge multiple Keras history dictionaries sequentially.
    """
    merged: Dict[str, List[float]] = {}
    for history in history_parts:
        for key, values in history.items():
            merged.setdefault(key, [])
            merged[key].extend([float(v) for v in values])
    return merged


def plot_transfer_history(
    history: Dict,
    accuracy_path: Path,
    loss_path: Path,
    model_label: str,
    dpi: int = 150,
) -> None:
    """
    Save accuracy and loss curves for a transfer-learning model.
    """
    accuracy_path = Path(accuracy_path)
    loss_path = Path(loss_path)
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    loss_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.get("val_accuracy", []), label="val_accuracy")
    plt.title(f"{model_label} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title(f"{model_label} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def infer_deployment_comment(backbone_name: str) -> str:
    """
    Simple qualitative deployment note.
    """
    backbone_name = backbone_name.lower()

    if backbone_name in {"mobilenet", "mobilenetv2"}:
        return "Highly deployment-friendly and lightweight."
    if backbone_name == "efficientnetb0":
        return "Reasonably lightweight with a good balance of efficiency and accuracy."
    if backbone_name == "resnet50":
        return "Stronger feature extractor but heavier for edge deployment."

    return "Deployment suitability should be verified after benchmarking."


def estimate_speed_bucket(backbone_name: str) -> str:
    """
    Qualitative speed label for report trade-off notes.
    """
    backbone_name = backbone_name.lower()

    if backbone_name in {"mobilenet", "mobilenetv2"}:
        return "fast"
    if backbone_name == "efficientnetb0":
        return "medium"
    if backbone_name == "resnet50":
        return "slower"

    return "unknown"


def train_one_transfer_model(
    dataset_root: Path,
    project_root: Path,
    training_config: Dict,
    model_name: str,
    model_config: Dict,
    logger=None,
) -> Dict:
    """
    Train a single transfer-learning backbone through:
    1. frozen-head training
    2. optional fine-tuning
    3. evaluation
    4. artifact saving
    """
    logger = logger or get_logger(name=f"transfer_learning_{model_name}")

    datasets = build_transfer_datasets(
        dataset_root=dataset_root,
        image_size=training_config["image_size"],
        batch_size=training_config["batch_size"],
        label_mode=training_config["label_mode"],
        seed=training_config["seed"],
        backbone_name=model_config["backbone"],
    )

    class_names = datasets["class_names"]
    num_classes = len(class_names)

    model, base_model = build_transfer_learning_model(
        backbone_name=model_config["backbone"],
        input_shape=model_config["input_shape"],
        weights=model_config["weights"],
        include_top=model_config["include_top"],
        pooling=model_config["pooling"],
        dense_units=model_config["dense_units"],
        dropout_rate=model_config["dropout_rate"],
        l2_regularization=model_config["l2_regularization"],
        num_classes=num_classes,
        label_mode=training_config["label_mode"],
        freeze_backbone=model_config.get("freeze_backbone", True),
    )

    artifact_cfg = model_config["artifacts"]
    model_dir = project_root / artifact_cfg["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / artifact_cfg["best_model_name"]
    final_model_path = model_dir / artifact_cfg["final_model_name"]
    history_path = model_dir / artifact_cfg["history_name"]
    metrics_path = model_dir / artifact_cfg["metrics_name"]
    accuracy_curve_path = project_root / artifact_cfg["accuracy_curve_path"]
    loss_curve_path = project_root / artifact_cfg["loss_curve_path"]

    compile_transfer_model(
        model=model,
        learning_rate=training_config["frozen_learning_rate"],
        optimizer_name=training_config["optimizer"],
        label_mode=training_config["label_mode"],
        num_classes=num_classes,
    )

    callbacks = build_training_callbacks(
        callback_config=model_config["callbacks"],
        best_model_path=best_model_path,
    )

    logger.info("Starting frozen-head training for %s...", model_name)
    start_time = time.perf_counter()

    frozen_history_obj = model.fit(
        datasets["train"],
        validation_data=datasets["valid"],
        epochs=training_config["epochs_frozen"],
        callbacks=callbacks,
        verbose=1,
    )

    history_parts = [frozen_history_obj.history]
    frozen_epochs_completed = len(frozen_history_obj.history.get("loss", []))

    fine_tune_cfg = model_config.get("fine_tune", {})
    finetune_epochs_completed = 0

    if fine_tune_cfg.get("enabled", False):
        logger.info("Starting fine-tuning for %s...", model_name)

        unfreeze_top_layers(
            base_model=base_model,
            n_layers=fine_tune_cfg.get("unfreeze_top_layers", 0),
        )

        compile_transfer_model(
            model=model,
            learning_rate=training_config["finetune_learning_rate"],
            optimizer_name=training_config["optimizer"],
            label_mode=training_config["label_mode"],
            num_classes=num_classes,
        )

        finetune_history_obj = model.fit(
            datasets["train"],
            validation_data=datasets["valid"],
            initial_epoch=frozen_epochs_completed,
            epochs=frozen_epochs_completed + training_config["epochs_finetune"],
            callbacks=callbacks,
            verbose=1,
        )

        history_parts.append(finetune_history_obj.history)
        finetune_epochs_completed = len(finetune_history_obj.history.get("loss", []))

    total_training_time_seconds = float(time.perf_counter() - start_time)
    history = merge_history_dicts(history_parts)

    logger.info("Saving final model for %s...", model_name)
    model.save(final_model_path)

    validation_metrics = evaluate_model_on_dataset(
        model=model,
        dataset=datasets["valid"],
        class_names=class_names,
        label_mode=training_config["label_mode"],
    )

    test_metrics = evaluate_model_on_dataset(
        model=model,
        dataset=datasets["test"],
        class_names=class_names,
        label_mode=training_config["label_mode"],
    )

    best_epoch = (
        int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i]) + 1)
        if history.get("val_loss")
        else None
    )

    metrics_payload = {
        "model_name": model_name,
        "backbone": model_config["backbone"],
        "class_names": class_names,
        "frozen_epochs_completed": frozen_epochs_completed,
        "finetune_epochs_completed": finetune_epochs_completed,
        "best_epoch": best_epoch,
        "training_time_seconds": total_training_time_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "speed_bucket": estimate_speed_bucket(model_config["backbone"]),
        "deployment_note": infer_deployment_comment(model_config["backbone"]),
    }

    save_json(history, history_path)
    save_json(metrics_payload, metrics_path)

    plot_transfer_history(
        history=history,
        accuracy_path=accuracy_curve_path,
        loss_path=loss_curve_path,
        model_label=model_name,
    )

    return {
        "model_name": model_name,
        "backbone": model_config["backbone"],
        "history": history,
        "metrics": metrics_payload,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "history_path": history_path,
        "metrics_path": metrics_path,
        "accuracy_curve_path": accuracy_curve_path,
        "loss_curve_path": loss_curve_path,
    }


def build_transfer_summary_table(results: List[Dict]) -> pd.DataFrame:
    """
    Build a concise comparison table for all trained transfer-learning models.
    """
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "model_name": metrics["model_name"],
                "backbone": metrics["backbone"],
                "val_accuracy": metrics["validation"]["accuracy"],
                "val_f1_score": metrics["validation"]["f1_score"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_f1_score": metrics["test"]["f1_score"],
                "training_time_seconds": metrics["training_time_seconds"],
                "speed_bucket": metrics["speed_bucket"],
                "deployment_note": metrics["deployment_note"],
                "best_model_path": metrics["best_model_path"],
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["val_accuracy", "test_accuracy"],
            ascending=False,
        ).reset_index(drop=True)
    return summary_df


def build_transfer_learning_report(summary_df: pd.DataFrame) -> str:
    """
    Create a markdown report covering accuracy, speed, and deployment trade-offs.
    """
    lines = [
        "# Transfer Learning Report",
        "",
        "## Overview",
        "",
        "This phase compares multiple pretrained backbones for aerial object classification using a two-stage strategy: frozen-head training followed by optional fine-tuning of the upper backbone layers.",
        "",
    ]

    if summary_df.empty:
        lines.extend([
            "No transfer-learning models were trained.",
        ])
        return "\n".join(lines)

    best_row = summary_df.iloc[0]

    lines.extend([
        "## Best Model So Far",
        "",
        f"- Best model: **{best_row['model_name']}**",
        f"- Validation accuracy: **{best_row['val_accuracy']:.4f}**",
        f"- Test accuracy: **{best_row['test_accuracy']:.4f}**",
        f"- Validation F1-score: **{best_row['val_f1_score']:.4f}**",
        f"- Training time (seconds): **{best_row['training_time_seconds']:.2f}**",
        "",
        "## Model Trade-Offs",
        "",
    ])

    for _, row in summary_df.iterrows():
        lines.extend([
            f"### {row['model_name']}",
            f"- Backbone: {row['backbone']}",
            f"- Validation accuracy: {row['val_accuracy']:.4f}",
            f"- Test accuracy: {row['test_accuracy']:.4f}",
            f"- Training time (seconds): {row['training_time_seconds']:.2f}",
            f"- Speed bucket: {row['speed_bucket']}",
            f"- Deployment note: {row['deployment_note']}",
            "",
        ])

    lines.extend([
        "## Comparison Table",
        "",
        summary_df.to_markdown(index=False),
        "",
    ])

    return "\n".join(lines)


def train_transfer_learning_suite(
    dataset_root: Path,
    project_root: Path,
    transfer_learning_config: Dict,
    logger=None,
) -> Dict:
    """
    Train all enabled transfer-learning models, save comparison artifacts,
    and write a report.
    """
    logger = logger or get_logger(name="transfer_learning_suite")

    training_config = transfer_learning_config["training"]
    model_registry = transfer_learning_config["models"]
    reporting_cfg = transfer_learning_config["reporting"]

    results = []

    for model_name, model_config in model_registry.items():
        if not model_config.get("enabled", False):
            logger.info("Skipping disabled model: %s", model_name)
            continue

        logger.info("Training transfer-learning model: %s", model_name)
        result = train_one_transfer_model(
            dataset_root=dataset_root,
            project_root=project_root,
            training_config=training_config,
            model_name=model_name,
            model_config=model_config,
            logger=logger,
        )
        results.append(result)

    summary_df = build_transfer_summary_table(results)

    comparison_table_path = project_root / reporting_cfg["comparison_table_path"]
    report_path = project_root / reporting_cfg["report_path"]

    save_dataframe(summary_df, comparison_table_path, index=False)
    save_text(build_transfer_learning_report(summary_df), report_path)

    return {
        "results": results,
        "summary_df": summary_df,
        "comparison_table_path": comparison_table_path,
        "report_path": report_path,
    }