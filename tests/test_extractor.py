from app.research.extractor import ContentExtractor
from app.models.schema import FetchedPage

def test_extractor_fallback():
    extractor = ContentExtractor()
    page = FetchedPage(
        url="http://example.com",
        status_code=200,
        content_type="text/html",
        html="<html><body><p>" + "This is a test paragraph with enough words to pass the minimum threshold of twenty words. " * 4 + "</p></body></html>"
    )
    doc = extractor.extract(page)
    assert doc.extraction_successful is True
    assert "test paragraph" in doc.text

def test_extractor_failure_on_short_content():
    extractor = ContentExtractor()
    page = FetchedPage(
        url="http://example.com",
        status_code=200,
        content_type="text/html",
        html="<html><body><p>Too short</p></body></html>"
    )
    doc = extractor.extract(page)
    assert doc.extraction_successful is False
