import sqlite3
import os
from contextlib import contextmanager
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

def get_db_path() -> str:
    return settings.database_path

def init_db():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                rolling_summary TEXT
            )
        ''')
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN rolling_summary TEXT")
        except sqlite3.OperationalError:
            pass

        
        # messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        # research_turns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                user_query TEXT NOT NULL,
                search_queries_json TEXT NOT NULL,
                opened_urls_json TEXT NOT NULL,
                final_answer TEXT,
                citations_json TEXT,
                retrieved_chunks_json TEXT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        try:
            cursor.execute("ALTER TABLE research_turns ADD COLUMN citations_json TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE research_turns ADD COLUMN retrieved_chunks_json TEXT")
        except sqlite3.OperationalError:
            pass
        
        # sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                turn_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                domain TEXT NOT NULL,
                fetched_at TIMESTAMP NOT NULL,
                extraction_status TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id),
                FOREIGN KEY (turn_id) REFERENCES research_turns (id)
            )
        ''')


        # chunks table (Stage 2)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                turn_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                embedding_id INTEGER,
                FOREIGN KEY (source_id) REFERENCES sources (id),
                FOREIGN KEY (session_id) REFERENCES sessions (id),
                FOREIGN KEY (turn_id) REFERENCES research_turns (id)
            )
        ''')

        conn.commit()
    logger.info(f"Database initialized at {db_path}")

@contextmanager
def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
