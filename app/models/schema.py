from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone

class SearchResult(BaseModel):
    url: str
    title: str
    content: Optional[str] = None
    raw_score: Optional[float] = None

class FetchedPage(BaseModel):
    url: str
    status_code: int
    content_type: str
    html: Optional[str] = None
    error: Optional[str] = None

class ExtractedDocument(BaseModel):
    url: str
    title: str
    domain: str
    text: str
    word_count: int
    extraction_successful: bool
    error: Optional[str] = None

class Session(BaseModel):
    id: Optional[int] = None
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Message(BaseModel):
    id: Optional[int] = None
    session_id: int
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ResearchTurn(BaseModel):
    id: Optional[int] = None
    session_id: int
    user_query: str
    search_queries_json: str
    opened_urls_json: str
    final_answer: Optional[str] = None
    citations_json: Optional[str] = None
    retrieved_chunks_json: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SourceRecord(BaseModel):
    id: Optional[int] = None
    session_id: int
    turn_id: int
    url: str
    title: str
    domain: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_status: str

# ----- Stage 2 models -----

class DocumentChunk(BaseModel):
    """A single chunk produced by the chunker from an ExtractedDocument."""
    chunk_id: Optional[int] = None   # SQLite rowid after persistence
    source_id: Optional[int] = None  # FK → sources.id
    session_id: Optional[int] = None
    turn_id: Optional[int] = None
    source_url: str
    title: str
    domain: str
    chunk_index: int
    text: str
    token_count: int
    embedding_id: Optional[int] = None  # position in FAISS index

class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever with its relevance score."""
    chunk_id: int
    source_url: str
    title: str
    domain: str
    text: str
    score: float
    citation_label: str = ""  # e.g. "[S1]" — filled in by context builder

class Citation(BaseModel):
    label: str      # e.g. "[S1]"
    title: str
    url: str
    domain: str


from dataclasses import dataclass

@dataclass
class PipelineEvent:
    stage: str
    message: str
    data: Optional[dict] = None

