import requests
from bs4 import BeautifulSoup
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebScraper:

    def scrape(self, url: str) -> dict:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(separator=" ")

            logger.info(f"Scraped URL: {url}")

            return {
                "text": " ".join(text.split()),
                "metadata": {
                    "source": url
                }
            }

        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            raise