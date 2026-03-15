"""
tune.py

# Goal
Hyperparameter tuning with correct CV and best model saving.

# Inputs
- configs/model_grid.yaml
- data/processed/X_train, y_train

# Outputs
- reports/model_results/tuning_results.csv
- models/best_model.joblib

# Steps
1) Load training split.
2) Load grids from yaml.
3) Build pipeline (preprocessor + model [+ sampler]).
4) Run GridSearchCV or RandomizedSearchCV.
5) Save cv_results_.
6) Save best estimator artifact.
7) Print best params and best score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


SearchType = Literal["grid", "random"]


def load_yaml_config(config_path: str | Path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding = "utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"YAML config must load as a dictionary: {config_path}")

    return config


def load_model_param_grid(
    model_name: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """
    Load parameter grid for a given model from YAML config.

    Supported YAML structures
    -------------------------
    1) Top-level:
       random_forest:
         param_grid:
           model__n_estimators: [100, 200]

    2) Nested under grids:
       grids:
         random_forest:
           model__n_estimators: [100, 200]

    3) Nested under grids with param_grid:
       grids:
         random_forest:
           param_grid:
             model__n_estimators: [100, 200]
    """
    config = load_yaml_config(config_path)

    # Case 1: top-level model entry
    if model_name in config:
        model_block = config[model_name]

        if not isinstance(model_block, dict):
            raise ValueError(f"Model '{model_name}' must be a dictionary.")

        if "param_grid" in model_block:
            param_grid = model_block["param_grid"]
        elif "model_params" in model_block:
            param_grid = model_block["model_params"]
        else:
            param_grid = model_block

    # Case 2: nested under "grids"
    elif "grids" in config and model_name in config["grids"]:
        model_block = config["grids"][model_name]

        if not isinstance(model_block, dict):
            raise ValueError(f"Model '{model_name}' inside 'grids' must be a dictionary.")

        if "param_grid" in model_block:
            param_grid = model_block["param_grid"]
        elif "model_params" in model_block:
            param_grid = model_block["model_params"]
        else:
            param_grid = model_block

    else:
        raise KeyError(f"Model '{model_name}' not found in config: {config_path}")

    if not isinstance(param_grid, dict):
        raise ValueError(f"Parameter grid for model '{model_name}' must be a dictionary.")

    return param_grid


def ensure_directory(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_search(
        *,
        pipeline,
        param_grid: dict[str, Any],
        cv,
        scoring: str = "f1_macro",
        search_type: SearchType = "grid",
        n_iter: int = 20,
        n_jobs: int = -1,
        refit: bool = True,
        verbose: int = 1,
        random_state: int = 42,
        return_train_score: bool = True
):
    if search_type == "grid":
        return GridSearchCV(
            estimator = pipeline,
            param_grid = param_grid,
            scoring = scoring,
            cv = cv,
            n_jobs = n_jobs,
            refit = refit,
            verbose = verbose,
            return_train_score = return_train_score
        )

    if search_type == "random":
        return RandomizedSearchCV(
            estimator = pipeline,
            param_distributions = param_grid,
            scoring = scoring,
            cv = cv,
            n_jobs = n_jobs,
            refit = refit,
            verbose = verbose,
            return_train_score = return_train_score
        )

    raise ValueError("search_type must be one of ['grid', 'random'].")


def save_cv_results(search, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_csv(output_path, index = False)

    return results_df


def save_best_params(search, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    best_info = {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_index": int(search.best_index_)
    }

    with output_path.open("w", encoding = "utf-8") as f:
        yaml.safe_dump(best_info, f, sort_keys = False)

    return best_info


def save_best_estimator(
    search,
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, output_path)
    return output_path


def run_tuning(
        *,
        model_name: str,
        pipeline,
        X,
        y,
        cv,
        config_path: str | Path,
        artifacts_dir: str | Path,
        scoring: str = "f1_macro",
        search_type: SearchType = "grid",
        n_iter: int = 20,
        n_jobs: int = -1,
        refit: bool = True,
        verbose: int = 1,
        random_state: int = 42,
        return_train_score: bool = True
):
    param_grid = load_model_param_grid(model_name = model_name, config_path = config_path)

    search = build_search(
        pipeline = pipeline,
        param_grid = param_grid,
        cv = cv,
        scoring = scoring,
        search_type = search_type,
        n_iter = n_iter,
        n_jobs = n_jobs,
        refit = refit,
        verbose = verbose,
        random_state = random_state,
        return_train_score = return_train_score
    )

    search.fit(X, y)

    artifacts_dir = ensure_directory(artifacts_dir)

    cv_results_path = artifacts_dir/f"{model_name}_cv_results.csv"
    best_params_path = artifacts_dir/f"{model_name}_best_params.yaml"
    best_estimator_path = artifacts_dir/f"{model_name}_best_estimator.joblib"

    cv_results_df = save_cv_results(search, cv_results_path)
    best_info = save_best_params(search, best_params_path)
    save_best_estimator(search, best_estimator_path)

    return {
        "model_name": model_name,
        "search": search,
        "best_estimator": search.best_estimator_,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_results_df": cv_results_df,
        "cv_results_path": cv_results_path,
        "best_params_path": best_params_path,
        "best_estimator_path": best_estimator_path,
        "saved_best_info": best_info
    }