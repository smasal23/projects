class Summarizer:

    @staticmethod
    def extract(data: dict) -> str:
        return data.get("summary", "")