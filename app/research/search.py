import httpx
from typing import List
from app.models.schema import SearchResult
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class TavilyClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key if api_key is not None else settings.tavily_api_key
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not self.api_key:
            logger.warning("No Tavily API key provided, returning empty results or mocking.")
            # For testing without real API key, return empty list or raise error
            return []
            
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, headers=headers, json=payload, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                results = []
                seen_urls = set()
                
                for idx, item in enumerate(data.get("results", [])):
                    url = item.get("url")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(SearchResult(
                            url=url,
                            title=item.get("title", ""),
                            content=item.get("content", ""),
                            raw_score=item.get("score", float(idx))
                        ))
                return results
            except Exception as e:
                logger.error(f"Tavily search failed for query '{query}': {e}")
                return []
