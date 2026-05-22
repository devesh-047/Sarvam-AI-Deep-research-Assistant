import pytest
import os
import sqlite3
from app.memory.db import init_db, get_connection
from app.memory.repositories import SessionRepository
from app.core.config import settings

def test_db_initialization(tmp_path):
    settings.database_path = str(tmp_path / "test.db")
    init_db()
    assert os.path.exists(settings.database_path)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "sessions" in tables
        assert "messages" in tables
        assert "research_turns" in tables
        assert "sources" in tables

def test_session_repository(tmp_path):
    settings.database_path = str(tmp_path / "test.db")
    init_db()
    repo = SessionRepository()
    session = repo.create("Test Session")
    assert session.id is not None
    assert session.title == "Test Session"
