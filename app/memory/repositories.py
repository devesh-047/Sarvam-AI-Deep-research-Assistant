import sqlite3
from typing import List, Optional
from datetime import datetime, timezone
from app.memory.db import get_connection
from app.models.schema import Session, Message, ResearchTurn, SourceRecord, DocumentChunk

class SessionRepository:
    def create(self, title: str) -> Session:
        with get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc)
            now_str = now.isoformat()
            cursor.execute(
                "INSERT INTO sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now_str, now_str)
            )
            conn.commit()
            return Session(id=cursor.lastrowid, title=title, created_at=now, updated_at=now)

    def list_all(self) -> List[Session]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [Session(id=r["id"], title=r["title"], created_at=datetime.fromisoformat(r["created_at"]), updated_at=datetime.fromisoformat(r["updated_at"])) for r in rows]

    def get_by_id(self, session_id: int) -> Optional[Session]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Session(id=row["id"], title=row["title"], created_at=datetime.fromisoformat(row["created_at"]), updated_at=datetime.fromisoformat(row["updated_at"]))

    def update_summary(self, session_id: int, summary: str) -> None:
        with get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "UPDATE sessions SET rolling_summary = ?, updated_at = ? WHERE id = ?",
                (summary, now_str, session_id)
            )
            conn.commit()

    def get_summary(self, session_id: int) -> Optional[str]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rolling_summary FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return row["rolling_summary"] if row else None

class MessageRepository:
    def create(self, session_id: int, role: str, content: str) -> Message:
        with get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc)
            now_str = now.isoformat()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now_str)
            )
            conn.commit()
            return Message(id=cursor.lastrowid, session_id=session_id, role=role, content=content, created_at=now)

class ResearchTurnRepository:
    def create(self, session_id: int, user_query: str, search_queries_json: str, opened_urls_json: str) -> ResearchTurn:
        with get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc)
            now_str = now.isoformat()
            cursor.execute(
                "INSERT INTO research_turns (session_id, user_query, search_queries_json, opened_urls_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_query, search_queries_json, opened_urls_json, now_str)
            )
            conn.commit()
            return ResearchTurn(
                id=cursor.lastrowid,
                session_id=session_id,
                user_query=user_query,
                search_queries_json=search_queries_json,
                opened_urls_json=opened_urls_json,
                created_at=now
            )

    def update_final_answer(self, turn_id: int, final_answer: str) -> None:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE research_turns SET final_answer = ? WHERE id = ?",
                (final_answer, turn_id)
            )
            conn.commit()

    def update_turn_results(self, turn_id: int, final_answer: str, citations_json: str, retrieved_chunks_json: str) -> None:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE research_turns SET final_answer = ?, citations_json = ?, retrieved_chunks_json = ? WHERE id = ?",
                (final_answer, citations_json, retrieved_chunks_json, turn_id)
            )
            conn.commit()

    def get_recent_turns(self, session_id: int, limit: int) -> List[ResearchTurn]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, session_id, user_query, search_queries_json, opened_urls_json, final_answer, citations_json, retrieved_chunks_json, created_at FROM research_turns WHERE session_id = ? AND final_answer IS NOT NULL ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            turns = []
            for r in rows:
                turns.append(ResearchTurn(
                    id=r["id"],
                    session_id=r["session_id"],
                    user_query=r["user_query"],
                    search_queries_json=r["search_queries_json"],
                    opened_urls_json=r["opened_urls_json"],
                    final_answer=r["final_answer"],
                    citations_json=r["citations_json"],
                    retrieved_chunks_json=r["retrieved_chunks_json"],
                    created_at=datetime.fromisoformat(r["created_at"])
                ))
            # Return in chronological order
            turns.reverse()
            return turns

    def get_all_for_session(self, session_id: int) -> List[ResearchTurn]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, session_id, user_query, search_queries_json, opened_urls_json, final_answer, citations_json, retrieved_chunks_json, created_at FROM research_turns WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            )
            rows = cursor.fetchall()
            turns = []
            for r in rows:
                turns.append(ResearchTurn(
                    id=r["id"],
                    session_id=r["session_id"],
                    user_query=r["user_query"],
                    search_queries_json=r["search_queries_json"],
                    opened_urls_json=r["opened_urls_json"],
                    final_answer=r["final_answer"],
                    citations_json=r["citations_json"],
                    retrieved_chunks_json=r["retrieved_chunks_json"],
                    created_at=datetime.fromisoformat(r["created_at"])
                ))
            return turns

            
class SourceRepository:
    def create(self, session_id: int, turn_id: int, url: str, title: str, domain: str, extraction_status: str) -> SourceRecord:
        with get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc)
            now_str = now.isoformat()
            cursor.execute(
                "INSERT INTO sources (session_id, turn_id, url, title, domain, fetched_at, extraction_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, turn_id, url, title, domain, now_str, extraction_status)
            )
            conn.commit()
            return SourceRecord(
                id=cursor.lastrowid,
                session_id=session_id,
                turn_id=turn_id,
                url=url,
                title=title,
                domain=domain,
                extraction_status=extraction_status,
                fetched_at=now
            )


class ChunkRepository:
    def insert(self, chunk: DocumentChunk) -> DocumentChunk:
        """Insert a single chunk. Returns the chunk with chunk_id set."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO chunks
                   (source_id, session_id, turn_id, chunk_text, chunk_index, token_count, embedding_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk.source_id, chunk.session_id, chunk.turn_id,
                 chunk.text, chunk.chunk_index, chunk.token_count, chunk.embedding_id)
            )
            conn.commit()
            chunk.chunk_id = cursor.lastrowid
            return chunk

    def bulk_insert(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Insert multiple chunks in a single transaction."""
        with get_connection() as conn:
            cursor = conn.cursor()
            for chunk in chunks:
                cursor.execute(
                    """INSERT INTO chunks
                       (source_id, session_id, turn_id, chunk_text, chunk_index, token_count, embedding_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk.source_id, chunk.session_id, chunk.turn_id,
                     chunk.text, chunk.chunk_index, chunk.token_count, chunk.embedding_id)
                )
                chunk.chunk_id = cursor.lastrowid
            conn.commit()
        return chunks

    def update_embedding_id(self, chunk_id: int, embedding_id: int) -> None:
        """Record the FAISS embedding_id for an already-inserted chunk."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chunks SET embedding_id = ? WHERE id = ?",
                (embedding_id, chunk_id)
            )
            conn.commit()

    def get_by_ids(self, chunk_ids: List[int]) -> List[DocumentChunk]:
        """Retrieve chunk rows (with source metadata joined) by a list of chunk ids."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT c.id, c.source_id, c.session_id, c.turn_id,
                           c.chunk_text, c.chunk_index, c.token_count, c.embedding_id,
                           s.url, s.title, s.domain
                    FROM chunks c
                    JOIN sources s ON c.source_id = s.id
                    WHERE c.id IN ({placeholders})""",
                chunk_ids
            )
            rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append(DocumentChunk(
                chunk_id=row["id"],
                source_id=row["source_id"],
                session_id=row["session_id"],
                turn_id=row["turn_id"],
                text=row["chunk_text"],
                chunk_index=row["chunk_index"],
                token_count=row["token_count"],
                embedding_id=row["embedding_id"],
                source_url=row["url"],
                title=row["title"],
                domain=row["domain"],
            ))
        return result

