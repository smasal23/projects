class RiskAnalyzer:

    @staticmethod
    def extract(data: dict) -> list:
        risks = []

        sentiment = data.get("sentiment_risk", {})

        for r in sentiment.get("risk_flags", []):
            risks.append({
                "risk": r,
                "severity": "medium",  # default or logic-based
                "description": r
            })

        return risks