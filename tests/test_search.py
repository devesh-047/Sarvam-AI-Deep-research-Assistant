import pytest
from app.research.search import TavilyClient
from app.models.schema import SearchResult
import json

@pytest.mark.asyncio
async def test_search_no_api_key():
    client = TavilyClient(api_key="")
    results = await client.search("test")
    assert results == []

# Real API testing would require mocking httpx.AsyncClient
