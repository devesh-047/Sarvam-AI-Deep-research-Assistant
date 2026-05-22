import httpx
import asyncio
from typing import List
from app.models.schema import FetchedPage
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class AsyncFetcher:
    def __init__(self):
        self.timeout = settings.fetch_timeout_seconds
        self.max_concurrent = settings.max_concurrent_fetches

    async def fetch_page(self, client: httpx.AsyncClient, url: str) -> FetchedPage:
        try:
            response = await client.get(url, follow_redirects=True)
            content_type = response.headers.get("Content-Type", "")
            return FetchedPage(
                url=str(response.url),
                status_code=response.status_code,
                content_type=content_type,
                html=response.text if "text/html" in content_type else None,
                error=None if response.status_code == 200 else f"Status: {response.status_code}"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return FetchedPage(
                url=url,
                status_code=0,
                content_type="",
                html=None,
                error=str(e)
            )

    async def fetch_all(self, urls: List[str]) -> List[FetchedPage]:
        sem = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_with_sem(client: httpx.AsyncClient, url: str):
            async with sem:
                return await self.fetch_page(client, url)

        limits = httpx.Limits(max_keepalive_connections=self.max_concurrent, max_connections=self.max_concurrent)
        timeout = httpx.Timeout(self.timeout)
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            tasks = [fetch_with_sem(client, url) for url in urls]
            return await asyncio.gather(*tasks)
