import os
from typing import List, Dict
from transformers import pipeline
from src.utils.logger import get_logger
from src.utils.config import CONFIG
from src.utils.helpers import save_json, load_json

logger = get_logger(__name__)


class TopicClassifier:
    def __init__(self):
        self.model_name = CONFIG["classification"]["model"]
        self.labels = CONFIG["classification"]["labels"]
        self.batch_size = CONFIG["classification"].get("batch_size", 8)
        self.threshold = CONFIG["classification"].get("confidence_threshold", 0.5)

        logger.info(f"Loading model: {self.model_name}")

        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=-1  # change to 0 if GPU
        )

    def classify_batch(self, texts: List[str]) -> List[Dict]:

        results = self.classifier(
            texts,
            candidate_labels=[
                "Eligibility of candidates (age, education, requirements)",
                "Financial benefits (stipend, payment, funding)",
                "Implementation process (how the scheme works)",
                "Stakeholder responsibilities (roles of companies, government)",
                "Application process (how candidates apply)",
                "Monitoring and evaluation (tracking performance)",
                "Scheme overview (general description of scheme)"
            ],
            hypothesis_template="This government regulation section is about {}.",
            truncation=True
        )

        return results

    def process_chunks(self, chunks: List[Dict]) -> List[Dict]:

        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Processing {len(texts)} chunks...")

        raw_results = self.classify_batch(texts)

        processed_results = []

        for chunk, result in zip(chunks, raw_results):

            # 🔥 MULTI-LABEL EXTRACTION
            top_labels = [
                {"label": l, "score": round(s, 4)}
                for l, s in zip(result["labels"], result["scores"])
                if s > 0.4
            ]

            # fallback if nothing passes threshold
            if not top_labels:
                top_labels = [{
                    "label": result["labels"][0],
                    "score": round(result["scores"][0], 4)
                }]

            # primary label (for compatibility)
            primary_label = top_labels[0]["label"]
            primary_score = top_labels[0]["score"]

            processed_results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],

                # 🔥 keep both
                "label": primary_label,
                "confidence": primary_score,

                "top_labels": top_labels  # 🔥 NEW FIELD
            })

        return processed_results

    def run(self, input_path: str, output_path: str):

        data = load_json(input_path)

        results = self.process_chunks(data)

        save_json(results, output_path)

        logger.info(f"Saved results: {output_path}")

        return results