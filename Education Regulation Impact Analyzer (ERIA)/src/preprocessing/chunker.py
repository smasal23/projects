import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    def __init__(self, chunk_size=800, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_sentences(self, text):
        return re.split(r'(?<=[.!?])\s+', text)

    def get_overlap(self, text):
        overlap_text = text[-self.overlap:]
        parts = re.split(r'(?<=[.!?])\s+', overlap_text)
        return parts[-1] if len(parts) > 1 else overlap_text

    def chunk(self, text):
        sentences = self.split_sentences(text)

        chunks = []
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:

            if len(current_chunk) + len(sentence) > self.chunk_size:

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip()
                })

                logger.info(f"Chunk {chunk_id} created")

                # 🔥 Correct overlap handling
                if chunk_id > 0:
                    overlap_text = self.get_overlap(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

                chunk_id += 1

            else:
                current_chunk += " " + sentence

        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip()
            })

        logger.info(f"Total chunks created: {len(chunks)}")

        return chunks