import pytest
from app.research.fetcher import AsyncFetcher
import httpx

@pytest.mark.asyncio
async def test_fetch_error_handling():
    fetcher = AsyncFetcher()
    # Assuming this URL fails or returns 404
    results = await fetcher.fetch_all(["http://localhost:9999/nonexistent"])
    assert len(results) == 1
    assert results[0].error is not None
