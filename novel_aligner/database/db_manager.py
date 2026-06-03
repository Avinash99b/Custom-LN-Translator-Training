"""
Database Module

SQLite-based storage for novels, chapters, embeddings, and alignments.
Provides efficient querying and caching capabilities.
"""
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
import json
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)


# SQL Schema
SCHEMA = """
-- Novels table
CREATE TABLE IF NOT EXISTS novels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    root_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    jp_chapter_count INTEGER DEFAULT 0,
    en_chapter_count INTEGER DEFAULT 0,
    metadata JSON
);

-- Chapters table
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    novel_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    language TEXT NOT NULL CHECK(language IN ('jp', 'en')),
    chapter_number INTEGER,
    title TEXT,
    content_hash TEXT,
    content_preview TEXT,
    fingerprint_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (novel_id) REFERENCES novels(id) ON DELETE CASCADE,
    UNIQUE(novel_id, relative_path)
);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    embedding_mode TEXT NOT NULL,
    vector BLOB,  -- Stored as serialized numpy array or JSON
    dimension INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE,
    UNIQUE(chapter_id, model_name, embedding_mode)
);

-- Relations (alignments) table
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    novel_id INTEGER NOT NULL,
    jp_chapter_id INTEGER NOT NULL,
    en_chapter_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    alignment_method TEXT DEFAULT 'sequence_dp',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (novel_id) REFERENCES novels(id) ON DELETE CASCADE,
    FOREIGN KEY (jp_chapter_id) REFERENCES chapters(id) ON DELETE CASCADE,
    FOREIGN KEY (en_chapter_id) REFERENCES chapters(id) ON DELETE CASCADE,
    UNIQUE(jp_chapter_id, en_chapter_id)
);

-- Validation results table
CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    novel_id INTEGER NOT NULL,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    matched_count INTEGER,
    unmatched_jp_count INTEGER,
    unmatched_en_count INTEGER,
    average_confidence REAL,
    details_json JSON,
    FOREIGN KEY (novel_id) REFERENCES novels(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chapters_novel_id ON chapters(novel_id);
CREATE INDEX IF NOT EXISTS idx_chapters_language ON chapters(language);
CREATE INDEX IF NOT EXISTS idx_chapters_path ON chapters(relative_path);
CREATE INDEX IF NOT EXISTS idx_embeddings_chapter_id ON embeddings(chapter_id);
CREATE INDEX IF NOT EXISTS idx_relations_novel_id ON relations(novel_id);
CREATE INDEX IF NOT EXISTS idx_relations_jp_chapter ON relations(jp_chapter_id);
CREATE INDEX IF NOT EXISTS idx_relations_en_chapter ON relations(en_chapter_id);
"""


class DatabaseManager:
    """
    Manages SQLite database operations for the novel aligner.
    
    Provides CRUD operations for novels, chapters, embeddings, and relations.
    """
    
    def __init__(self, db_path: str = "novel_aligner.db", wal_mode: bool = True):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file
            wal_mode: Enable WAL mode for better concurrent reads
        """
        self.db_path = Path(db_path)
        self.wal_mode = wal_mode
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize database schema."""
        with self.get_connection() as conn:
            # Enable WAL mode if requested
            if self.wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
            
            # Create tables
            conn.executescript(SCHEMA)
            conn.commit()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Novel operations
    
    def create_novel(
        self,
        name: str,
        root_path: str,
        jp_count: int = 0,
        en_count: int = 0,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Create or update a novel record.
        
        Returns:
            Novel ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO novels (name, root_path, jp_chapter_count, en_chapter_count, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    root_path = excluded.root_path,
                    jp_chapter_count = excluded.jp_chapter_count,
                    en_chapter_count = excluded.en_chapter_count,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (name, root_path, jp_count, en_count, json.dumps(metadata) if metadata else None),
            )
            
            novel_id = cursor.lastrowid
            if novel_id is None:
                # Get existing ID
                row = conn.execute(
                    "SELECT id FROM novels WHERE name = ?", (name,)
                ).fetchone()
                novel_id = row['id']
            
            conn.commit()
            return novel_id
    
    def get_novel(self, novel_id: int) -> Optional[Dict[str, Any]]:
        """Get novel by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM novels WHERE id = ?", (novel_id,)
            ).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_novel_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get novel by name."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM novels WHERE name = ?", (name,)
            ).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def list_novels(self) -> List[Dict[str, Any]]:
        """List all novels."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM novels ORDER BY updated_at DESC").fetchall()
            return [dict(row) for row in rows]
    
    def delete_novel(self, novel_id: int) -> None:
        """Delete a novel and all related data."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM novels WHERE id = ?", (novel_id,))
            conn.commit()
    
    # Chapter operations
    
    def create_chapter(
        self,
        novel_id: int,
        file_path: str,
        relative_path: str,
        language: str,
        chapter_number: Optional[int] = None,
        title: Optional[str] = None,
        content_hash: Optional[str] = None,
        content_preview: Optional[str] = None,
        fingerprint_json: Optional[Dict] = None,
    ) -> int:
        """Create a chapter record."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chapters (
                    novel_id, file_path, relative_path, language,
                    chapter_number, title, content_hash, content_preview, fingerprint_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(novel_id, relative_path) DO UPDATE SET
                    file_path = excluded.file_path,
                    chapter_number = excluded.chapter_number,
                    title = excluded.title,
                    content_hash = excluded.content_hash,
                    content_preview = excluded.content_preview,
                    fingerprint_json = excluded.fingerprint_json
                """,
                (
                    novel_id, file_path, relative_path, language,
                    chapter_number, title, content_hash, content_preview,
                    json.dumps(fingerprint_json) if fingerprint_json else None,
                ),
            )
            
            chapter_id = cursor.lastrowid
            if chapter_id is None:
                row = conn.execute(
                    "SELECT id FROM chapters WHERE novel_id = ? AND relative_path = ?",
                    (novel_id, relative_path),
                ).fetchone()
                chapter_id = row['id']
            
            conn.commit()
            return chapter_id
    
    def get_chapters_by_novel(
        self,
        novel_id: int,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all chapters for a novel."""
        with self.get_connection() as conn:
            if language:
                rows = conn.execute(
                    "SELECT * FROM chapters WHERE novel_id = ? AND language = ? ORDER BY chapter_number ASC, relative_path ASC",
                    (novel_id, language),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM chapters WHERE novel_id = ? ORDER BY language, chapter_number ASC, relative_path ASC",
                    (novel_id,),
                ).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_chapter(self, chapter_id: int) -> Optional[Dict[str, Any]]:
        """Get chapter by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE id = ?", (chapter_id,)
            ).fetchone()
            
            if row:
                result = dict(row)
                if result.get('fingerprint_json'):
                    result['fingerprint_json'] = json.loads(result['fingerprint_json'])
                return result
            return None
    
    # Embedding operations
    
    def save_embedding(
        self,
        chapter_id: int,
        model_name: str,
        embedding_mode: str,
        vector: List[float],
        dimension: int,
    ) -> int:
        """Save an embedding vector."""
        import numpy as np
        
        # Serialize as JSON for portability
        vector_json = json.dumps(vector)
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO embeddings (chapter_id, model_name, embedding_mode, vector, dimension)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chapter_id, model_name, embedding_mode) DO UPDATE SET
                    vector = excluded.vector,
                    dimension = excluded.dimension,
                    created_at = CURRENT_TIMESTAMP
                """,
                (chapter_id, model_name, embedding_mode, vector_json, dimension),
            )
            
            conn.commit()
            return cursor.lastrowid
    
    def get_embedding(
        self,
        chapter_id: int,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get embedding for a chapter."""
        with self.get_connection() as conn:
            if model_name:
                row = conn.execute(
                    """
                    SELECT * FROM embeddings 
                    WHERE chapter_id = ? AND model_name = ?
                    """,
                    (chapter_id, model_name),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM embeddings WHERE chapter_id = ?",
                    (chapter_id,),
                ).fetchone()
            
            if row:
                result = dict(row)
                if result.get('vector'):
                    result['vector'] = json.loads(result['vector'])
                return result
            return None
    
    # Relation operations
    
    def create_relation(
        self,
        novel_id: int,
        jp_chapter_id: int,
        en_chapter_id: int,
        confidence: float,
        alignment_method: str = 'sequence_dp',
    ) -> int:
        """Create a relation between chapters."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO relations (novel_id, jp_chapter_id, en_chapter_id, confidence, alignment_method)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(jp_chapter_id, en_chapter_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    alignment_method = excluded.alignment_method
                """,
                (novel_id, jp_chapter_id, en_chapter_id, confidence, alignment_method),
            )
            
            conn.commit()
            return cursor.lastrowid
    
    def save_relations_batch(
        self,
        novel_id: int,
        relations: List[Tuple[int, int, float]],
        alignment_method: str = 'sequence_dp',
    ) -> None:
        """Save multiple relations in a batch."""
        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO relations (novel_id, jp_chapter_id, en_chapter_id, confidence, alignment_method)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(jp_chapter_id, en_chapter_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    alignment_method = excluded.alignment_method
                """,
                [(novel_id, jp_id, en_id, conf, alignment_method) for jp_id, en_id, conf in relations],
            )
            conn.commit()
    
    def get_relations_by_novel(self, novel_id: int) -> List[Dict[str, Any]]:
        """Get all relations for a novel."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT r.*, 
                       jp.relative_path as jp_path, jp.title as jp_title,
                       en.relative_path as en_path, en.title as en_title
                FROM relations r
                JOIN chapters jp ON r.jp_chapter_id = jp.id
                JOIN chapters en ON r.en_chapter_id = en.id
                WHERE r.novel_id = ?
                ORDER BY r.confidence DESC
                """,
                (novel_id,),
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    # Validation operations
    
    def save_validation_result(
        self,
        novel_id: int,
        matched: int,
        unmatched_jp: int,
        unmatched_en: int,
        avg_confidence: float,
        details: Optional[Dict] = None,
    ) -> int:
        """Save validation results."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO validation_results (
                    novel_id, matched_count, unmatched_jp_count, unmatched_en_count,
                    average_confidence, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (novel_id, matched, unmatched_jp, unmatched_en, avg_confidence, json.dumps(details) if details else None),
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_latest_validation(self, novel_id: int) -> Optional[Dict[str, Any]]:
        """Get latest validation result for a novel."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM validation_results WHERE novel_id = ? ORDER BY run_timestamp DESC LIMIT 1",
                (novel_id,),
            ).fetchone()
            
            if row:
                result = dict(row)
                if result.get('details_json'):
                    result['details_json'] = json.loads(result['details_json'])
                return result
            return None
    
    # Statistics
    
    def get_statistics(self, novel_id: int) -> Dict[str, Any]:
        """Get statistics for a novel."""
        with self.get_connection() as conn:
            # Chapter counts
            jp_count = conn.execute(
                "SELECT COUNT(*) FROM chapters WHERE novel_id = ? AND language = 'jp'",
                (novel_id,),
            ).fetchone()[0]
            
            en_count = conn.execute(
                "SELECT COUNT(*) FROM chapters WHERE novel_id = ? AND language = 'en'",
                (novel_id,),
            ).fetchone()[0]
            
            # Relation counts
            relation_count = conn.execute(
                "SELECT COUNT(*) FROM relations WHERE novel_id = ?",
                (novel_id,),
            ).fetchone()[0]
            
            # Average confidence
            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM relations WHERE novel_id = ?",
                (novel_id,),
            ).fetchone()[0]
            
            return {
                'novel_id': novel_id,
                'jp_chapters': jp_count,
                'en_chapters': en_count,
                'total_relations': relation_count,
                'average_confidence': avg_conf or 0.0,
            }
