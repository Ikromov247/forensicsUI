"""SQLite database for storing detection results."""
import sqlite3
import json
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Optional

from config import DATABASE_DIR


@dataclass
class DetectionRecord:
    """A detected object with tracking info."""
    obj_id: int
    class_id: int
    class_name: str
    confidence: float
    similarity: float
    frame_ids: List[int]
    bboxes: List[List[int]]
    features: Optional[np.ndarray] = None


class Database:
    """Simple SQLite database for forensics results."""

    def __init__(self, name: str = "forensics"):
        self.db_path = DATABASE_DIR / f"{name}.db"
        self._init_tables()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    obj_id INTEGER PRIMARY KEY,
                    class_id INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    similarity REAL,
                    frame_ids TEXT,
                    bboxes TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    obj_id INTEGER PRIMARY KEY,
                    features BLOB,
                    FOREIGN KEY (obj_id) REFERENCES objects (obj_id)
                )
            """)

    def save_detection(self, record: DetectionRecord):
        """Save a detection record to database."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO objects
                (obj_id, class_id, class_name, confidence, similarity, frame_ids, bboxes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.obj_id,
                record.class_id,
                record.class_name,
                record.confidence,
                record.similarity,
                json.dumps(record.frame_ids),
                json.dumps(record.bboxes)
            ))

            if record.features is not None:
                conn.execute("""
                    INSERT OR REPLACE INTO features (obj_id, features)
                    VALUES (?, ?)
                """, (record.obj_id, record.features.tobytes()))

    def get_all_detections(self) -> List[DetectionRecord]:
        """Get all detection records."""
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT obj_id, class_id, class_name, confidence, similarity, frame_ids, bboxes
                FROM objects ORDER BY similarity DESC
            """)
            records = []
            for row in cursor:
                records.append(DetectionRecord(
                    obj_id=row[0],
                    class_id=row[1],
                    class_name=row[2],
                    confidence=row[3],
                    similarity=row[4],
                    frame_ids=json.loads(row[5]),
                    bboxes=json.loads(row[6])
                ))
            return records

    def get_top_matches(self, limit: int = 10) -> List[DetectionRecord]:
        """Get top N matches by similarity."""
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT obj_id, class_id, class_name, confidence, similarity, frame_ids, bboxes
                FROM objects ORDER BY similarity DESC LIMIT ?
            """, (limit,))
            records = []
            for row in cursor:
                records.append(DetectionRecord(
                    obj_id=row[0],
                    class_id=row[1],
                    class_name=row[2],
                    confidence=row[3],
                    similarity=row[4],
                    frame_ids=json.loads(row[5]),
                    bboxes=json.loads(row[6])
                ))
            return records

    def clear(self):
        """Clear all records."""
        with self._connect() as conn:
            conn.execute("DELETE FROM features")
            conn.execute("DELETE FROM objects")
