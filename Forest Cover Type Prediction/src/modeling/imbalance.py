"""
imbalance.py

# Goal
Provide imbalance-handling strategies safely inside training/CV pipelines.

# Rules (very important)
- If using SMOTE/resampling, it MUST be inside Pipeline and inside CV folds.
- Never resample before train-test split.

# Steps
1) Provide class_weight strategies for compatible models.
2) Provide SMOTE pipeline builder (imblearn.pipeline.Pipeline).
3) Provide helper:
   - make_pipeline(preprocessor, model, sampler=None)
"""

from __future__ import annotations

from typing import Any, Literal

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone


ImbalanceStrategy = Literal["none", "class_weight", "smote"]


def estimator_supports_class_weight(estimator: BaseEstimator):
    return "class_weight" in estimator.get_params(deep = True)

def apply_class_weight(estimator: BaseEstimator, class_weight: str | dict[int, float] = "balanced"):
    if not estimator_supports_class_weight(estimator):
        raise ValueError(
            f"{estimator.__class__.__name__} does not support `class_weight`."
        )

    estimator_copy = clone(estimator)
    estimator_copy.set_params(class_weight = class_weight)
    return estimator_copy


def build_smote(
        *,
        random_state: int = 42,
        k_neighbors: int = 5,
        sampling_strategy: str | float | dict[int, int] = "auto"
):
    return SMOTE(
        random_state = random_state,
        k_neighbors = k_neighbors,
        sampling_strategy = sampling_strategy
    )


def build_imbalanced_pipeline(
        *,
        preprocessor: BaseEstimator,
        estimator: BaseEstimator,
        strategy: ImbalanceStrategy = "none",
        class_weight: str | dict[int, float] = "balanced",
        smote_k_neighbors: int = 5,
        smote_sampling_strategy: str | float | dict[int, int] = "auto",
        random_state: int = 42
):
    estimator_copy = clone(estimator)
    preprocessor_copy = clone(preprocessor)

    if strategy == "none":
        return ImbPipeline(
            steps = [
                ("preprocessor", preprocessor_copy),
                ("model", estimator_copy)
            ]
        )

    if strategy == "class_weight":
        weighted_estimator = apply_class_weight(
            estimator = estimator_copy,
            class_weight = class_weight
        )
        return ImbPipeline(
            steps = [
                ("preprocessor", preprocessor_copy),
                ("model", weighted_estimator)
            ]
        )

    if strategy == "smote":
        smote = build_smote(
            random_state = random_state,
            k_neighbors = smote_k_neighbors,
            sampling_strategy = smote_sampling_strategy
        )
        return ImbPipeline(
            steps = [
                ("preprocessor", preprocessor_copy),
                ("smote", smote),
                ("model", estimator_copy)
            ]
        )

    raise ValueError(
        f"Unknown strategy = {strategy!r}. Choose from"
        f"['none', 'class_weight', 'smote']."
    )


def get_imbalance_config(
        strategy: ImbalanceStrategy,
        *,
        class_weight: str | dict[int, float] = "balanced",
        smote_k_neighbors: int = 5,
        smote_sampling_strategy: str | float | dict[int, int] = "auto"
):
    config: dict[str, Any] = {"strategy": strategy}

    if strategy == "class_weight":
        config["class_weight"] = class_weight

    if strategy == "smote":
        config["smote_k_neighbors"] = smote_k_neighbors
        config["smote_sampling_strategy"] = smote_sampling_strategy

    return config