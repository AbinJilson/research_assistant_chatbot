import logging
import requests

# Web scraping (optional)
try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebScraper:
    async def scrape(self, url: str):
        if not CRAWL4AI_AVAILABLE:
            logger.warning("crawl4ai not installed. Falling back to basic requests.")
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.error(f"Error fetching URL {url} with requests: {e}")
                return None

        try:
            crawler = AsyncWebCrawler()
            result = await crawler.arun(url=url)
            if result and result.content:
                return result.content
            logger.warning(f"crawl4ai returned no content for {url}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url} with crawl4ai: {e}")
            return None
