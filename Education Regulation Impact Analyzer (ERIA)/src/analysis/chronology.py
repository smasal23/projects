class ChronologyAnalyzer:

    @staticmethod
    def extract(data: dict) -> dict:
        return data.get("impact_assessment", {})