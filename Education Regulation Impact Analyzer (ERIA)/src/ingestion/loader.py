import os
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.validator import DocumentValidator
from src.utils.helpers import save_json, truncate_text
from src.utils.config import PATHS, CONFIG
from src.utils.logger import get_logger
from src.utils.config import BASE_DIR

logger = get_logger(__name__)


class DataLoader:
    def __init__(self):
        self.extractor = PDFExtractor()

    def load_pdf(self, file_path: str) -> dict:
        """
        Full pipeline:
        Validate → Extract → Clean → Save
        """
        try:
            # Validate
            DocumentValidator.validate_file(file_path)

            # Extract
            data = self.extractor.extract(file_path)

            # Basic cleaning
            cleaned_text = self.basic_cleaning(data["text"])

            cleaned_text = cleaned_text  # no truncation

            output = {
                "text": cleaned_text,
                "metadata": data["metadata"]
            }

            # Save
            file_name = os.path.basename(file_path).replace(".pdf", ".json")
            save_path = os.path.join(BASE_DIR, "data", "interim", file_name)

            save_json(output, save_path)

            logger.info(f"Saved interim data: {save_path}")

            return output

        except Exception as e:
            logger.error(f"Loading failed: {e}")
            raise

    def basic_cleaning(self, text: str) -> str:
        """
        Minimal cleaning (as per constraints)
        """
        text = text.replace("\n", " ")
        text = " ".join(text.split())  # remove extra spaces
        return text