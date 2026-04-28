import os
import json
from datetime import datetime

from src.preprocessing.cleaner import remove_noise_lines, normalize_text, remove_short_sentences, filter_chunks
from src.preprocessing.chunker import TextChunker
from src.utils.logger import get_logger
from src.utils.config import BASE_DIR

logger = get_logger(__name__)


class PreprocessingPipeline:
    def __init__(self):
        self.chunker = TextChunker()

    def process(self, input_file: str) -> dict:
        try:
            logger.info(f"Processing file: {input_file}")

            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = data["text"]

            logger.info(f"Original text length: {len(text)}")

            # 🔥 STEP 1: REMOVE OCR NOISE
            text = remove_noise_lines(text)

            # 🔥 STEP 2: NORMALIZE
            text = normalize_text(text)

            # 🔥 STEP 3: FILTER
            text = remove_short_sentences(text)

            logger.info(f"Cleaned text length: {len(text)}")

            # 🔥 STEP 3: CHUNKING
            chunks = self.chunker.chunk(text)

            # STEP 5 🔥 FILTER BAD CHUNKS
            chunks = filter_chunks(chunks)

            metadata = {
                **data["metadata"],
                "processed_at": datetime.utcnow().isoformat(),
                "num_chunks": len(chunks),
                "text_length": len(text)
            }

            output = {
                "metadata": metadata,
                "chunks": chunks
            }

            # 🔥 SAVE OUTPUTS
            file_name = os.path.basename(input_file)

            processed_path = os.path.join(BASE_DIR, "data", "processed", file_name)
            chunks_path = os.path.join(BASE_DIR, "artifacts", "chunks", file_name)

            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            os.makedirs(os.path.dirname(chunks_path), exist_ok=True)

            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)

            logger.info("Preprocessing completed successfully")

            return output

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise