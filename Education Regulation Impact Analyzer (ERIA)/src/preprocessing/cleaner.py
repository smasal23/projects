import re
from src.utils.logger import get_logger

logger = get_logger(__name__)

WORD_PATTERN = re.compile(r'^[A-Za-z]{3,}$')


def is_valid_word(word: str) -> bool:
    if not WORD_PATTERN.match(word):
        return False

    if len(set(word)) <= 2:
        return False

    return True


def line_quality_score(line: str) -> float:
    words = line.split()

    if not words:
        return 0

    valid_words = sum(1 for w in words if is_valid_word(w))
    return valid_words / len(words)


def remove_noise_lines(text: str) -> str:
    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if len(line) < 40:
            continue

        # Symbol noise filter
        bad_chars = sum(1 for c in line if not c.isalnum() and c not in " .,:/-()")
        if bad_chars / len(line) > 0.3:
            continue

        # Word quality filter
        if line_quality_score(line) < 0.5:
            continue

        # Repetition noise
        if re.search(r'(.)\1{4,}', line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_short_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)

    return " ".join([
        s for s in sentences
        if len(s.split()) > 6
    ])


def normalize_text(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9.,:;()/%\-\s]', ' ', text)
    text = re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def filter_chunks(chunks):
    return [
        c for c in chunks
        if len(c["text"].split()) > 8
    ]