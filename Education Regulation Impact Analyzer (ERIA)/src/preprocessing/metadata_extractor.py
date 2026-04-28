from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataExtractor:

    @staticmethod
    def enrich(metadata: dict, text: str) -> dict:
        logger.info("Enriching metadata")

        metadata["processed_at"] = datetime.utcnow().isoformat()
        metadata["text_length"] = len(text)
        metadata["num_chunks_estimated"] = max(1, len(text) // 1000)

        logger.debug(f"Metadata updated: {metadata}")

        return metadata