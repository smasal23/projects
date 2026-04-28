from groq import Groq
from src.utils.config import GROQ_API_KEY, CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GroqClient:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = CONFIG["llm"]["model"]

    def generate(self, prompt: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=CONFIG["llm"]["temperature"],
                max_tokens=CONFIG["llm"]["max_tokens"]
            )

            return response.choices[0].message.content


        except Exception as e:

            if "rate_limit" in str(e).lower():
                logger.error("Rate limit hit. Skipping retries.")

                return ""

            logger.error(f"Groq LLM error: {e}")

            return ""