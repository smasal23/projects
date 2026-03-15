"""
models.py

# Goal
Central factory to create baseline and tuned models.

# Steps
1) Implement get_model(name) to return unfitted estimator.
2) Include:
   - LogisticRegression
   - KNN
   - DecisionTree
   - RandomForest
   - XGBoost (optional)
3) Ensure random_state where applicable.
4) Keep defaults simple for baseline stage.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

default_random_state = 42


# def get_model(name: str): ..
def get_baseline_models(random_state: int = default_random_state):
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter = 1000,
            solver = "lbfgs",
            class_weight = None,
            n_jobs = None,
            random_state = random_state
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf = 1,
            class_weight = None,
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators = 300,
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf = 1,
            max_features = "sqrt",
            class_weight = None,
            n_jobs = 1,
            random_state=random_state
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight=None,
            n_jobs=1,
            random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators = 100,
            learning_rate = 0.1,
            max_depth = 3,
            random_state=random_state
        ),
        "knn": KNeighborsClassifier(
            n_neighbors = 5,
            weights = "uniform",
            metric = "minkowski",
            p = 2
        ),
        "gaussian_nb": GaussianNB(),
        "svc_rbf": SVC(
            C = 1.0,
            kernel = "rbf",
            gamma = "scale",
            probability = False,
            class_weight = None,
            random_state = random_state
        )
    }

    return models


def get_model_names(random_state: int = default_random_state):
    return list(get_baseline_models(random_state = random_state).keys())


def get_single_model(model_name: str, random_state: int = default_random_state):
    models = get_baseline_models(random_state = random_state)

    if model_name not in models:
        available = ", ".join(models.keys())
        raise KeyError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    return models[model_name]


def main():
    models = get_baseline_models()

    print("=" * 60)
    print("Available baseline models")
    print("=" * 60)

    for name, model in models.items():
        print(f"{name}: {model}")


if __name__ == "__main__":
    main()