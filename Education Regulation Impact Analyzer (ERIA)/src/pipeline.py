import os
from typing import Dict, Any, List

from src.ingestion.loader import DataLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.analysis.classifier import TopicClassifier

from src.llm.client import GroqClient
from src.llm.prompts import build_prompt
from src.llm.parser import LLMParser

from src.analysis.summarizer import Summarizer
from src.analysis.stakeholder import StakeholderAnalyzer
from src.analysis.risk_analyzer import RiskAnalyzer
from src.analysis.chronology import ChronologyAnalyzer

from src.utils.helpers import load_json, save_json
from src.utils.logger import get_logger
from src.utils.config import BASE_DIR


logger = get_logger(__name__)


class ERIAPipeline:

    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = PreprocessingPipeline()
        self.classifier = TopicClassifier()
        self.llm = GroqClient()

    # ----------------------------------
    # STEP 1: INGESTION
    # ----------------------------------
    def ingest(self, file_path: str) -> Dict:
        logger.info("Starting ingestion...")
        output = self.loader.load_pdf(file_path)
        return output

    # ----------------------------------
    # STEP 2: PREPROCESSING + CHUNKING
    # ----------------------------------
    def preprocess(self, input_file: str) -> Dict:
        logger.info("Starting preprocessing...")
        output = self.preprocessor.process(input_file)
        return output

    # ----------------------------------
    # STEP 3: CLASSIFICATION (OPTIONAL)
    # ----------------------------------
    def classify(self, chunks: List[Dict]) -> List[Dict]:
        logger.info("Starting classification...")
        results = self.classifier.process_chunks(chunks)
        return results

    # ----------------------------------
    # STEP 4: FILTER + BUILD LLM INPUT
    # ----------------------------------
    def build_llm_input(self, chunks: List[Dict]) -> str:
        logger.info("Building LLM input...")

        full_text = " ".join([c["text"] for c in chunks])

        keywords = [
            "eligibility", "criteria", "scheme", "implementation",
            "impact", "guidelines", "regulation", "requirements",
            "internship", "benefits", "conditions", "rules"
        ]

        filtered_lines = [
            line for line in full_text.split("\n")
            if any(k in line.lower() for k in keywords)
        ]

        filtered_text = "\n".join(filtered_lines)

        MAX_CHARS = 12000
        return filtered_text[:MAX_CHARS]

    # ----------------------------------
    # STEP 5: LLM EXTRACTION
    # ----------------------------------
    def run_llm(self, text: str) -> Dict:
        logger.info("Running LLM extraction...")

        prompt = build_prompt(text)
        response = self.llm.generate(prompt)

        print("\n🔍 RAW RESPONSE:\n", response[:500])

        # Fix common JSON issues
        response = response.replace("{{", "{").replace("}}", "}")

        parsed = LLMParser.parse(response)

        return parsed

    # ----------------------------------
    # STEP 6: POST ANALYSIS
    # ----------------------------------
    def post_process(self, data: Dict) -> Dict:
        logger.info("Running post-analysis...")

        return {
            "summary": Summarizer.extract(data),
            "stakeholders": StakeholderAnalyzer.extract(data),
            "risks": RiskAnalyzer.extract(data),
            "timeline": ChronologyAnalyzer.extract(data),
            "raw": data
        }

    # ----------------------------------
    # FULL PIPELINE
    # ----------------------------------
    def run(self, file_path: str) -> Dict[str, Any]:

        # 1. Ingest
        ingest_output = self.ingest(file_path)

        # Save interim (like notebook)
        interim_path = os.path.join(BASE_DIR, "data", "interim", "temp.json")
        save_json(ingest_output, interim_path)

        # 2. Preprocess
        processed = self.preprocess(interim_path)
        chunks = processed["chunks"]

        if not chunks:
            raise ValueError("No chunks generated — check preprocessing")

        # 3. (Optional) Classification
        # classified = self.classify(chunks)

        # 4. Build LLM Input
        llm_input = self.build_llm_input(chunks)

        if not llm_input.strip():
            raise ValueError("Empty LLM input — filtering too aggressive")

        # 5. Run LLM
        llm_output = self.run_llm(llm_input)

        if "error" in llm_output:
            raise ValueError(f"LLM failed: {llm_output}")

        # Save raw LLM output
        llm_path = os.path.join(BASE_DIR, "artifacts", "llm_outputs", "final.json")
        save_json(llm_output, llm_path)

        # 6. Post Analysis
        final_output = self.post_process(llm_output)

        return final_output