from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def build_optimizer(name: str = "adam", learning_rate: float = 1e-3) -> tf.keras.optimizers.Optimizer:
    name = name.lower()

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    raise ValueError(f"Unsupported optimizer: {name}")


def resolve_loss_and_metrics(
    label_mode: str = "binary",
    num_classes: int = 2,
    loss_name: str = "auto",
) -> Tuple[str, List[tf.keras.metrics.Metric]]:
    """
    Resolve compile-time loss and metrics.
    """
    if label_mode == "binary" or num_classes == 2:
        loss = "binary_crossentropy" if loss_name == "auto" else loss_name
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
        return loss, metrics

    loss = "categorical_crossentropy" if loss_name == "auto" else loss_name
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    ]
    return loss, metrics


def collect_true_labels_from_dataset(dataset, label_mode: str = "binary") -> np.ndarray:
    y_true = []

    for _, labels in dataset:
        label_array = labels.numpy()
        y_true.append(label_array)

    y_true = np.concatenate(y_true, axis=0)

    if label_mode == "binary":
        return y_true.reshape(-1).astype(int)

    if y_true.ndim > 1:
        return np.argmax(y_true, axis=1).astype(int)

    return y_true.astype(int)


def predict_label_indices(model, dataset, label_mode: str = "binary", threshold: float = 0.5) -> np.ndarray:
    y_prob = model.predict(dataset, verbose=0)

    if label_mode == "binary":
        return (y_prob.reshape(-1) >= threshold).astype(int)

    return np.argmax(y_prob, axis=1).astype(int)


def evaluate_classification_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names,
    label_mode: str = "binary",
) -> Dict:
    average = "binary" if label_mode == "binary" else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def evaluate_model_on_dataset(model, dataset, class_names, label_mode: str = "binary") -> Dict:
    y_true = collect_true_labels_from_dataset(dataset=dataset, label_mode=label_mode)
    y_pred = predict_label_indices(model=model, dataset=dataset, label_mode=label_mode)
    return evaluate_classification_predictions(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        label_mode=label_mode,
    )