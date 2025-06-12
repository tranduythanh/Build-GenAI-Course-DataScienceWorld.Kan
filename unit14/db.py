from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "toeic.db"


def init_db() -> None:
    """Create tables if they do not exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            image_path TEXT,
            user_answer TEXT,
            transcript TEXT,
            sample TEXT,
            score TEXT,
            feedback TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_session(
    image_path: str,
    user_answer: str,
    transcript: str | None,
    sample: str,
    score: str,
    feedback: str,
) -> None:
    """Save a practice session to the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        (
            "INSERT INTO sessions (ts, image_path, user_answer, "
            "transcript, sample, score, feedback) VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            datetime.utcnow().isoformat(),
            image_path,
            user_answer,
            transcript,
            sample,
            score,
            feedback,
        ),
    )
    conn.commit()
    conn.close()
