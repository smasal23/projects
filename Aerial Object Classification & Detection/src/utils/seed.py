import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Some runtimes may not support deterministic ops fully.
            pass
    except ImportError:
        # TensorFlow may not be installed in every environment.
        pass