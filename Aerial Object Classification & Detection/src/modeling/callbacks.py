from pathlib import Path
from typing import Dict, List

import tensorflow as tf


def build_training_callbacks(
    callback_config: Dict,
    best_model_path: Path,
) -> List[tf.keras.callbacks.Callback]:
    """
    Build training callbacks from config.
    """
    callbacks = []

    early_cfg = callback_config.get("early_stopping", {})
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=early_cfg.get("monitor", "val_loss"),
            mode=early_cfg.get("mode", "min"),
            patience=early_cfg.get("patience", 5),
            min_delta=early_cfg.get("min_delta", 0.0),
            restore_best_weights=early_cfg.get("restore_best_weights", True),
            verbose=1,
        )
    )

    ckpt_cfg = callback_config.get("model_checkpoint", {})
    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=ckpt_cfg.get("monitor", "val_loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_best_only=ckpt_cfg.get("save_best_only", True),
            verbose=1,
        )
    )

    reduce_cfg = callback_config.get("reduce_lr_on_plateau", {})
    if reduce_cfg.get("enabled", False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_cfg.get("monitor", "val_loss"),
                mode=reduce_cfg.get("mode", "min"),
                factor=reduce_cfg.get("factor", 0.5),
                patience=reduce_cfg.get("patience", 2),
                min_lr=reduce_cfg.get("min_lr", 1e-6),
                verbose=1,
            )
        )

    return callbacks