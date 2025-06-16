import logging
import requests
import asyncio
from typing import Optional, Dict, Any

# Web scraping (optional)
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.crawler import CrawlerConfig
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.crawler = None
        if CRAWL4AI_AVAILABLE:
            self._init_crawler()
    
    def _init_crawler(self):
        """Initialize the crawler with appropriate configuration"""
        try:
            config = CrawlerConfig(
                headless=True,
                browser="chromium",
                verbose=False
            )
            self.crawler = AsyncWebCrawler(config=config)
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {e}")
            self.crawler = None
    
    async def _scrape_with_playwright(self, url: str) -> Optional[str]:
        """Scrape using Playwright with proper async handling"""
        if not self.crawler:
            return None
            
        try:
            result = await self.crawler.arun(url=url)
            return result.content if result and result.content else None
        except Exception as e:
            logger.error(f"Error scraping {url} with Playwright: {e}")
            return None
    
    async def _scrape_with_requests(self, url: str) -> Optional[str]:
        """Fallback to requests if Playwright fails"""
        def _make_request(url: str) -> Optional[str]:
            try:
                response = requests.get(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0'}, 
                    timeout=10
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.error(f"Error fetching URL {url} with requests: {e}")
                return None
                
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _make_request, url)
    
    async def scrape(self, url: str) -> Optional[str]:
        """
        Scrape content from a URL with fallback mechanisms
        
        Args:
            url: The URL to scrape
            
        Returns:
            str: The scraped content, or None if all methods fail
        """
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            logger.error(f"Invalid URL: {url}")
            return None
            
        # Try with Playwright if available
        if CRAWL4AI_AVAILABLE and self.crawler:
            content = await self._scrape_with_playwright(url)
            if content:
                return content
                
        # Fallback to requests
        return await self._scrape_with_requests(url)
