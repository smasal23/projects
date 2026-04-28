#%%
import fitz
import pytesseract
from PIL import Image
import re
from datetime import datetime
from collections import Counter
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------
# Cleaning Utilities
# -----------------------------

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,:/()\-]", "", text)
    return text.strip()


def text_quality_score(text: str) -> float:
    if not text or len(text) < 50:
        return 0

    valid_chars = sum(c.isalnum() or c in " .,:/()-" for c in text)
    return valid_chars / len(text)


def is_text_valid(text: str) -> bool:
    return text_quality_score(text) > 0.6


def remove_repeated_lines(text: str) -> str:
    """
    Remove repeated headers/footers
    """
    lines = text.split("\n")
    counts = Counter(lines)

    cleaned = [line for line in lines if counts[line] < 5]
    return "\n".join(cleaned)


# -----------------------------
# Main Extractor
# -----------------------------

class PDFExtractor:
    def __init__(self):
        pass

    def extract(self, file_path: str) -> dict:
        try:
            doc = fitz.open(file_path)

            full_text = ""

            for page_num, page in enumerate(doc):

                # -----------------------------
                # 1. Layout-aware extraction
                # -----------------------------
                blocks = page.get_text("blocks")

                # 🔥 FIX: SORT BLOCKS (CRITICAL)
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

                pymupdf_text = " ".join(
                    block[4] for block in blocks if block[4].strip()
                )
                pymupdf_text = clean_text(pymupdf_text)

                # -----------------------------
                # 2. OCR fallback if needed
                # -----------------------------
                use_ocr = not is_text_valid(pymupdf_text)

                ocr_text = ""

                if use_ocr:
                    logger.warning(f"OCR fallback for page {page_num}")

                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples
                    )

                    ocr_text = pytesseract.image_to_string(
                        img,
                        config="--oem 3 --psm 4"
                    )

                    ocr_text = clean_text(ocr_text)

                # -----------------------------
                # 3. Hybrid selection
                # -----------------------------
                candidates = [pymupdf_text]

                if ocr_text:
                    candidates.append(ocr_text)
                    candidates.append(pymupdf_text + " " + ocr_text)

                best_text = max(candidates, key=text_quality_score)

                logger.info(
                    f"Page {page_num} | Scores → "
                    f"PDF: {text_quality_score(pymupdf_text):.2f}, "
                    f"OCR: {text_quality_score(ocr_text):.2f}"
                )

                full_text += best_text + "\n"

            # -----------------------------
            # 4. Post-processing cleanup
            # -----------------------------
            full_text = remove_repeated_lines(full_text)
            full_text = clean_text(full_text)

            metadata = {
                "file_name": file_path.split("/")[-1],
                "num_pages": len(doc),
                "extracted_at": datetime.utcnow().isoformat(),
                "text_length": len(full_text)
            }

            logger.info(f"Extraction complete: {file_path}")

            return {
                "text": full_text,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise