import logging
import os
from src.utils.config import CONFIG, PATHS
from src.utils.config import BASE_DIR


def get_logger(name="ERIA"):
    log_file = os.path.join(BASE_DIR, "logs", "app.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)

    level = CONFIG["logging"]["level"]
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger