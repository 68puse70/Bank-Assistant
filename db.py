import sqlite3
import datetime
import json
from indic_assistant.utils.config import Config
from indic_assistant.utils.logger import logger

class DatabaseManager:
    """Handles incremental persistence of assistant interactions and sessions using SQLite."""

    def __init__(self):
        self.db_path = Config.DB_PATH
        logger.info(f"Initializing SQLite database at: {self.db_path}")
        self._create_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    original_text TEXT,
                    english_translation TEXT,
                    llm_prompt TEXT,
                    llm_response TEXT,
                    translated_response TEXT,
                    latency_ms REAL,
                    confidence REAL,
                    errors TEXT,
                    retries INTEGER DEFAULT 0
                )
            """)
            
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    summary TEXT
                )
            """)
            
            conn.commit()
            logger.info("Database tables initialized successfully.")

    def start_session(self, session_id: str):
        """Creates a new session entry."""
        start_time = datetime.datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sessions (session_id, start_time) VALUES (?, ?)",
                    (session_id, start_time)
                )
                conn.commit()
                logger.info(f"Started session log: {session_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Session {session_id} already exists in database.")
        except Exception as e:
            logger.error(f"Failed to start session in database: {e}")

    def log_interaction(
        self,
        session_id: str,
        original_text: str,
        english_translation: str,
        llm_prompt: str,
        llm_response: str,
        translated_response: str,
        latency_ms: float,
        confidence: float = None,
        errors: str = None,
        retries: int = 0
    ):
        """Logs a single interaction turn."""
        timestamp = datetime.datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO interactions (
                        session_id, timestamp, original_text, english_translation, 
                        llm_prompt, llm_response, translated_response, 
                        latency_ms, confidence, errors, retries
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, timestamp, original_text, english_translation,
                    llm_prompt, llm_response, translated_response,
                    latency_ms, confidence, errors, retries
                ))
                conn.commit()
                logger.info("Log: Incremental interaction saved to database.")
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def end_session(self, session_id: str, summary: str):
        """Ends the session and saves the summary."""
        end_time = datetime.datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE sessions SET end_time = ?, summary = ? WHERE session_id = ?",
                    (end_time, summary, session_id)
                )
                conn.commit()
                logger.info(f"Ended session log: {session_id}")
        except Exception as e:
            logger.error(f"Failed to end session: {e}")

    def get_session_history(self, session_id: str):
        """Retrieves history of a session (for memory restoration if needed)."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp ASC",
                    (session_id,)
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []
