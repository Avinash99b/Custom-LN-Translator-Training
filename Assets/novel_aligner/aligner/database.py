import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional

class Database:
    def __init__(self, db_path: str = "aligner_cache.sqlite"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self):
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS novels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                novel_id INTEGER,
                language TEXT,
                file_path TEXT,
                title TEXT,
                summary TEXT,
                characters TEXT,
                events TEXT,
                UNIQUE(novel_id, language, file_path),
                FOREIGN KEY(novel_id) REFERENCES novels(id)
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                chapter_id INTEGER PRIMARY KEY,
                vector BLOB,
                FOREIGN KEY(chapter_id) REFERENCES chapters(id)
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                novel_id INTEGER,
                jp_chapter_path TEXT,
                en_chapter_path TEXT,
                confidence REAL,
                FOREIGN KEY(novel_id) REFERENCES novels(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chapters_novel_lang ON chapters(novel_id, language);
        """)
        self.conn.commit()

    def get_or_create_novel(self, name: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO novels (name) VALUES (?)", (name,))
        self.conn.commit()
        cursor.execute("SELECT id FROM novels WHERE name = ?", (name,))
        return cursor.fetchone()["id"]

    def save_fingerprint(self, novel_id: int, lang: str, file_path: str, fingerprint: Dict[str, Any]):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO chapters (novel_id, language, file_path, title, summary, characters, events)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (novel_id, lang, file_path, fingerprint.get("title", ""), 
              fingerprint.get("summary", ""), json.dumps(fingerprint.get("characters", [])), 
              json.dumps(fingerprint.get("events", []))))
        chapter_id = cursor.lastrowid or cursor.execute(
            "SELECT id FROM chapters WHERE novel_id=? AND language=? AND file_path=?", 
            (novel_id, lang, file_path)).fetchone()["id"]
        
        # Save vector as bytes
        vector = np.array(fingerprint["embedding"], dtype=np.float32).tobytes()
        cursor.execute("INSERT OR REPLACE INTO embeddings (chapter_id, vector) VALUES (?, ?)", (chapter_id, vector))
        self.conn.commit()

    def get_fingerprint(self, novel_id: int, lang: str, file_path: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.title, c.summary, c.characters, c.events, e.vector 
            FROM chapters c JOIN embeddings e ON c.id = e.chapter_id
            WHERE c.novel_id = ? AND c.language = ? AND c.file_path = ?
        """, (novel_id, lang, file_path))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "title": row["title"],
            "summary": row["summary"],
            "characters": json.loads(row["characters"]),
            "events": json.loads(row["events"]),
            "embedding": np.frombuffer(row["vector"], dtype=np.float32).tolist()
        }