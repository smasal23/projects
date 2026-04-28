import time
from typing import Dict, Any

from src.llm.client import GroqClient
from src.llm.prompts import build_prompt
from src.llm.parser import LLMParser
from src.utils.logger import get_logger

logger = get_logger(__name__)


# class ERIAOrchestrator:
#
#     def __init__(self):
#         self.llm = GroqClient()
#
#     # -----------------------------
#     # VALIDATION
#     # -----------------------------
#     def validate_schema(self, data: Dict) -> bool:
#         required_keys = [
#             "regulation_topic",
#             "chronology",
#             "impact_analysis",
#             "sentiment_risk",
#             "summary",
#             "stakeholder_report",
#             "impact_assessment",
#             "positives",
#             "negatives"
#         ]
#         return all(k in data for k in required_keys)
#
#     # -----------------------------
#     # HALLUCINATION CHECK
#     # -----------------------------
#     def detect_hallucination(self, data: Dict) -> bool:
#         text_blob = str(data).lower()
#
#         red_flags = [
#             "unknown",
#             "not mentioned",
#             "n/a",
#             "no data",
#             "lorem ipsum"
#         ]
#
#         return any(flag in text_blob for flag in red_flags)
#
#     # -----------------------------
#     # CORE LLM CALL
#     # -----------------------------
#     def extract_facts(self, text: str, retries=3):
#
#         for i in range(retries):
#             try:
#                 prompt = build_prompt(text)
#
#                 response = self.llm.generate(prompt)
#
#                 print("\n🔍 RAW RESPONSE:\n", response[:500])  # debug
#
#                 parsed = LLMParser.parse(response)
#
#                 # -----------------------------
#                 # 🔥 FALLBACK RETRY (PUT HERE)
#                 # -----------------------------
#                 if "error" in parsed:
#                     print("\n⚠️ Retrying with simplified prompt...")
#
#                     simple_prompt = f"""
#                     Extract structured policy info in JSON.
#
#                     Return JSON with keys:
#                     regulation_topic, chronology, impact_analysis
#
#                     Text:
#                     {text}
#                     """
#
#                     response = self.llm.generate(simple_prompt)
#
#                     print("\n🔁 FALLBACK RESPONSE:\n", response[:500])  # debug
#
#                     parsed = LLMParser.parse(response)
#
#                 # -----------------------------
#                 # VALIDATION
#                 # -----------------------------
#                 if "error" in parsed:
#                     raise ValueError(parsed["error"])
#
#                 if not self.validate_schema(parsed):
#                     raise ValueError("Schema validation failed")
#
#                 if self.detect_hallucination(parsed):
#                     logger.warning("Hallucination detected")
#
#                 return parsed
#
#             except Exception as e:
#                 wait = (2 ** i)
#                 logger.warning(f"Retry {i + 1}: {e} | waiting {wait}s")
#                 time.sleep(wait)
#
#         return {
#             "error": "stage1_failed",
#             "raw": text[:1000]
#         }
#
#     # -----------------------------
#     # PIPELINE ENTRY
#     # -----------------------------
#     def run(self, text: str) -> Dict:
#         return self.extract_facts(text)

class ERIAOrchestrator:

    def __init__(self):
        self.llm = GroqClient()

    def run(self, text: str) -> Dict:
        prompt = build_prompt(text)

        response = self.llm.generate(prompt)

        if not response:
            return {"error": "empty_llm_response"}

        print("\n🔍 RAW RESPONSE:\n", response[:500])

        # -----------------------------
        # BASIC CLEANING
        # -----------------------------
        response = response.replace("{{", "{").replace("}}", "}")
        response = response.replace("```json", "").replace("```", "").strip()

        # -----------------------------
        # PRIMARY PARSE
        # -----------------------------
        parsed = LLMParser.parse(response)

        # -----------------------------
        # FALLBACK (ONLY IF PARSE FAILS)
        # -----------------------------
        if isinstance(parsed, dict) and "error" in parsed:
            print("\n⚠️ Parser failed — retrying once...\n")

            retry_response = self.llm.generate(prompt)
            retry_response = retry_response.replace("{{", "{").replace("}}", "}")

            parsed = LLMParser.parse(retry_response)

        # -----------------------------
        # FINAL OUTPUT
        # -----------------------------
        return parsed