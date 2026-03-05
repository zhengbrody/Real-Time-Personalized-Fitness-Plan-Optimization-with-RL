#!/usr/bin/env python3
"""
Initialize database schema.
"""

import sqlite3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.config import DATABASE_URL


def init_database():
    """Initialize SQLite database with required tables."""
    
    # Parse SQLite URL
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
    else:
        db_path = "data/fitness.db"
    
    # Create directory if needed
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Training sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date DATE NOT NULL,
            action_id INTEGER NOT NULL,
            workout_type TEXT,
            intensity TEXT,
            duration_minutes INTEGER,
            completed BOOLEAN DEFAULT 0,
            rpe INTEGER,
            mood INTEGER,
            soreness INTEGER,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # User feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id INTEGER,
            action_id INTEGER NOT NULL,
            reward REAL,
            completion BOOLEAN,
            satisfaction INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
        )
    """)
    
    # Daily states table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_states (
            state_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date DATE NOT NULL,
            readiness_score REAL,
            sleep_score REAL,
            hrv REAL,
            resting_hr REAL,
            fatigue INTEGER,
            state_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, date)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_date ON training_sessions(user_id, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_states_user_date ON daily_states(user_id, date)")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized at: {db_path}")
    print("✅ Tables created: users, training_sessions, user_feedback, daily_states")


if __name__ == "__main__":
    init_database()

