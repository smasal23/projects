import os
import requests
from src.utils.helpers import ensure_dir
from src.utils.config import PATHS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_pdf(url: str, filename: str):
    try:
        ensure_dir(PATHS["data"]["raw"])

        save_path = os.path.join(PATHS["data"]["raw"], filename)

        response = requests.get(url)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded: {save_path}")

        return save_path

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    # Example
    url = "https://example.com/sample.pdf"
    download_pdf(url, "sample.pdf")