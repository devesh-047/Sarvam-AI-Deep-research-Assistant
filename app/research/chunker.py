"""
Token-aware document chunker with overlap.

Each chunk preserves full source metadata so no information is lost
when chunks travel through the pipeline independently.
"""
from typing import List
import tiktoken

from app.models.schema import ExtractedDocument, DocumentChunk
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Re-use the same encoding across calls (cl100k_base covers GPT-4 / most modern models)
_ENCODING = None

def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def _count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def chunk_document(
    doc: ExtractedDocument,
    source_id: int,
    session_id: int,
    turn_id: int,
    chunk_size: int = None,
    overlap: int = None,
) -> List[DocumentChunk]:
    """
    Split an ExtractedDocument into overlapping token-aware chunks.

    Args:
        doc: The extracted document to split.
        source_id: FK to the sources table row.
        session_id / turn_id: Propagated to every chunk.
        chunk_size: Max tokens per chunk. Defaults to settings.chunk_size_tokens.
        overlap: Overlap tokens between consecutive chunks. Defaults to settings.chunk_overlap_tokens.

    Returns:
        List of DocumentChunk objects (chunk_id is None until persisted).
    """
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size_tokens
    overlap = overlap if overlap is not None else settings.chunk_overlap_tokens

    if not doc.text or not doc.extraction_successful:
        logger.warning(f"Skipping chunking for failed/empty doc: {doc.url}")
        return []

    enc = _get_encoding()
    tokens = enc.encode(doc.text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return []

    chunks: List[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()

        if len(chunk_text.split()) >= 10:  # skip whitespace-only slivers
            chunks.append(DocumentChunk(
                source_id=source_id,
                session_id=session_id,
                turn_id=turn_id,
                source_url=doc.url,
                title=doc.title,
                domain=doc.domain,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(chunk_tokens),
            ))
            chunk_index += 1

        if end == total_tokens:
            break
        start = end - overlap  # slide back by overlap

    logger.info(f"Chunked '{doc.url}' → {len(chunks)} chunks (total {total_tokens} tokens)")
    return chunks
