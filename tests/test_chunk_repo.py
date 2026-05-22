"""Tests for chunk persistence (ChunkRepository)."""
import pytest
from app.core.config import settings
from app.memory.db import init_db
from app.memory.repositories import (
    SessionRepository, ResearchTurnRepository,
    SourceRepository, ChunkRepository
)
from app.models.schema import DocumentChunk
import json


@pytest.fixture(autouse=True)
def _setup_db(tmp_path):
    settings.database_path = str(tmp_path / "test.db")
    init_db()


def _create_session_turn_source():
    sess = SessionRepository().create("Test")
    turn = ResearchTurnRepository().create(
        session_id=sess.id,
        user_query="test query",
        search_queries_json=json.dumps(["test"]),
        opened_urls_json=json.dumps(["https://example.com"]),
    )
    src = SourceRepository().create(
        session_id=sess.id,
        turn_id=turn.id,
        url="https://example.com",
        title="Test",
        domain="example.com",
        extraction_status="success",
    )
    return sess, turn, src


def test_chunk_insert_and_retrieve():
    sess, turn, src = _create_session_turn_source()
    repo = ChunkRepository()

    chunk = DocumentChunk(
        source_id=src.id,
        session_id=sess.id,
        turn_id=turn.id,
        source_url="https://example.com",
        title="Test",
        domain="example.com",
        chunk_index=0,
        text="This is chunk text.",
        token_count=4,
    )
    saved = repo.insert(chunk)
    assert saved.chunk_id is not None

    retrieved = repo.get_by_ids([saved.chunk_id])
    assert len(retrieved) == 1
    assert retrieved[0].text == "This is chunk text."


def test_bulk_insert():
    sess, turn, src = _create_session_turn_source()
    repo = ChunkRepository()

    chunks = [
        DocumentChunk(
            source_id=src.id, session_id=sess.id, turn_id=turn.id,
            source_url="https://example.com", title="T", domain="example.com",
            chunk_index=i, text=f"chunk {i}", token_count=2,
        )
        for i in range(5)
    ]
    saved = repo.bulk_insert(chunks)
    assert all(c.chunk_id is not None for c in saved)


def test_update_embedding_id():
    sess, turn, src = _create_session_turn_source()
    repo = ChunkRepository()

    chunk = DocumentChunk(
        source_id=src.id, session_id=sess.id, turn_id=turn.id,
        source_url="https://example.com", title="T", domain="example.com",
        chunk_index=0, text="text", token_count=1,
    )
    saved = repo.insert(chunk)
    repo.update_embedding_id(saved.chunk_id, 42)

    rows = repo.get_by_ids([saved.chunk_id])
    assert rows[0].embedding_id == 42
