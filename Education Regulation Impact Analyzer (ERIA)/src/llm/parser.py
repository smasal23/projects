import json
import re


class LLMParser:

    @staticmethod
    def clean_json(text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()

        # Extract JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return ""

        text = match.group()

        # Fix common issues
        text = re.sub(r"(\w+):", r'"\1":', text)  # add quotes to keys
        text = text.replace("'", '"')            # replace single quotes

        return text

    REQUIRED_KEYS = [
        "regulation_topic",
        "chronology",
        "impact_analysis",
        "sentiment_risk",
        "summary",
        "stakeholder_report",
        "impact_assessment",
        "positives",
        "negatives"
    ]

    @staticmethod
    def validate_schema(self, data: dict) -> bool:
        required_keys = [
            "regulation_topic",
            "chronology",
            "impact_analysis",
            "summary",
            "stakeholder_report",
            "impact_assessment",
            "positives",
            "negatives"
        ]

        present = [k for k in required_keys if k in data]

        return len(present) >= 3  # allow partial success

    @staticmethod
    def parse(text: str):
        try:
            text = text.replace("```json", "").replace("```", "").strip()

            # Fix common issues
            text = text.replace("{{", "{").replace("}}", "}")

            # Extract JSON safely
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {"error": "no_json_found", "raw": text}

            json_str = match.group()

            # Try strict parse
            try:
                return json.loads(json_str)
            except:
                # Attempt fix for trailing commas
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)

                return json.loads(json_str)

        except Exception as e:
            return {"error": str(e), "raw": text}