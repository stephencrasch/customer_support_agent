import sqlite3
from datetime import datetime
import time
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database name constant
DATABASE_NAME = "msg_summary.db"

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        # Enable row factory for dictionary-like access
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_database() -> None:
    """Initialize the database tables with proper schema."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # SQLite uses INTEGER PRIMARY KEY AUTOINCREMENT, not SERIAL
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def save_summary_to_db(summary: str):
    """Save a summary to the database using the context manager."""
    try:
        # Ensure database is initialized before saving
        init_database()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # SQLite uses ? placeholders, not %s
            cursor.execute(
                "INSERT INTO conversation_summaries (summary) VALUES (?)",
                (summary,)
            )
            conn.commit()
            logger.info(f"Summary saved to database: {summary[:50]}...")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
        raise

def search_summaries_in_db(query: str):
    """Search for summaries in the database."""
    try:
        # Ensure database is initialized before searching
        init_database()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # SQLite uses LIKE instead of ILIKE (case-insensitive search)
            # Use LOWER() for case-insensitive matching
            cursor.execute(
                "SELECT summary FROM conversation_summaries WHERE LOWER(summary) LIKE LOWER(?) ORDER BY created_at DESC LIMIT 5",
                (f"%{query}%",)
            )
            results = cursor.fetchall()
            logger.info(f"Found {len(results)} matching summaries for query: {query}")
            return [row['summary'] for row in results]
    except Exception as e:
        logger.error(f"Failed to search summaries: {e}")
        return []