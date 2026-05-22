import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from app.models.schema import FetchedPage, ExtractedDocument
from app.core.logging import get_logger

logger = get_logger(__name__)

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except:
        return ""

class ContentExtractor:
    def extract(self, page: FetchedPage, default_title: str = "") -> ExtractedDocument:
        domain = extract_domain(page.url)
        
        if not page.html or page.error:
            return ExtractedDocument(
                url=page.url,
                title=default_title,
                domain=domain,
                text="",
                word_count=0,
                extraction_successful=False,
                error=page.error or "No HTML content"
            )

        # Try trafilatura first
        text = trafilatura.extract(page.html, include_links=False, include_images=False, include_tables=False)
        
        # Fallback to BeautifulSoup
        if not text or len(text.split()) < 20:
            logger.info(f"Trafilatura failed or returned short text for {page.url}, falling back to BeautifulSoup.")
            soup = BeautifulSoup(page.html, "html.parser")
            for script in soup(["script", "style", "nav", "footer"]):
                script.extract()
            text = soup.get_text(separator="\n")
            # simple cleanup
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(chunk for chunk in lines if chunk)
            
        word_count = len(text.split())
        success = word_count >= 50  # reject extremely low-content pages
        
        return ExtractedDocument(
            url=page.url,
            title=default_title,
            domain=domain,
            text=text,
            word_count=word_count,
            extraction_successful=success,
            error=None if success else "Content too short or extraction failed"
        )
