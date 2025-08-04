"""SQLite helper functions."""
import sqlite3
from pathlib import Path


def get_connection(db_path: str = "data/trades/trades.db") -> sqlite3.Connection:
    """Create database connection."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)
