"""
Similarity Matrix Module

Computes weighted similarity scores between Japanese and English chapters
using multiple signals:
- Embedding similarity (semantic)
- Title similarity (lexical/semantic)
- Character overlap
- Event overlap
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher

from .chapter_fingerprint import ChapterFingerprint
from .embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class SimilarityScore:
    """
    Detailed similarity score between two chapters.
    
    Attributes:
        total: Weighted total similarity score
        embedding_similarity: Semantic embedding similarity
        title_similarity: Title-based similarity
        character_overlap: Character name overlap score
        event_overlap: Event overlap score
        weights: Weights used for computation
    """
    total: float
    embedding_similarity: float = 0.0
    title_similarity: float = 0.0
    character_overlap: float = 0.0
    event_overlap: float = 0.0
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'embedding': 0.5,
                'title': 0.2,
                'character': 0.2,
                'event': 0.1,
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'embedding_similarity': self.embedding_similarity,
            'title_similarity': self.title_similarity,
            'character_overlap': self.character_overlap,
            'event_overlap': self.event_overlap,
            'weights': self.weights,
        }


class SimilarityMatrix:
    """
    Computes and stores similarity matrix between JP and EN chapters.
    
    The matrix has shape (n_jp, n_en) where each cell contains
    the weighted similarity score.
    """
    
    def __init__(
        self,
        jp_fingerprints: List[ChapterFingerprint],
        en_fingerprints: List[ChapterFingerprint],
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        """
        Initialize the similarity matrix.
        
        Args:
            jp_fingerprints: Japanese chapter fingerprints
            en_fingerprints: English chapter fingerprints
            embedding_engine: Engine for computing embedding similarities
        """
        self.jp_fingerprints = jp_fingerprints
        self.en_fingerprints = en_fingerprints
        self.embedding_engine = embedding_engine
        
        self.n_jp = len(jp_fingerprints)
        self.n_en = len(en_fingerprints)
        
        # Initialize matrix storage
        self._matrix: Optional[np.ndarray] = None
        self._detailed_matrix: List[List[SimilarityScore]] = []
    
    def compute(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Compute the full similarity matrix.
        
        Args:
            weights: Weights for different similarity components
            use_embeddings: Whether to use embedding similarity
            
        Returns:
            Similarity matrix of shape (n_jp, n_en)
        """
        if weights is None:
            weights = {
                'embedding': 0.5,
                'title': 0.2,
                'character': 0.2,
                'event': 0.1,
            }
        
        logger.info(f"Computing similarity matrix ({self.n_jp} x {self.n_en})")
        
        # Compute embedding similarity matrix if available
        embedding_sim = None
        if use_embeddings and self.embedding_engine:
            embedding_sim = self._compute_embedding_similarity()
        
        # Initialize detailed matrix
        self._detailed_matrix = []
        
        # Compute similarity for each pair
        rows = []
        for i, jp_fp in enumerate(self.jp_fingerprints):
            row = []
            for j, en_fp in enumerate(self.en_fingerprints):
                score = self._compute_pair_similarity(
                    jp_fp, en_fp, weights, embedding_sim[i, j] if embedding_sim is not None else None
                )
                row.append(score.total)
            rows.append(row)
            self._detailed_matrix.append(row)
        
        self._matrix = np.array(rows)
        logger.info(f"Similarity matrix computed. Range: [{self._matrix.min():.3f}, {self._matrix.max():.3f}]")
        
        return self._matrix
    
    def _compute_embedding_similarity(self) -> np.ndarray:
        """Compute embedding similarity matrix using vectorized operations."""
        if not self.embedding_engine:
            raise ValueError("No embedding engine available")
        
        # Extract embeddings
        jp_embeddings = [fp.embedding for fp in self.jp_fingerprints if fp.embedding is not None]
        en_embeddings = [fp.embedding for fp in self.en_fingerprints if fp.embedding is not None]
        
        if not jp_embeddings or not en_embeddings:
            logger.warning("Missing embeddings, computing them now...")
            # Would need to compute embeddings here
            raise ValueError("Missing embeddings")
        
        # Use vectorized similarity computation
        return self.embedding_engine.similarity_matrix(jp_embeddings, en_embeddings)
    
    def _compute_pair_similarity(
        self,
        jp_fp: ChapterFingerprint,
        en_fp: ChapterFingerprint,
        weights: Dict[str, float],
        precomputed_embedding_sim: Optional[float] = None,
    ) -> SimilarityScore:
        """Compute similarity between a single pair of chapters."""
        
        # Embedding similarity
        emb_sim = precomputed_embedding_sim
        if emb_sim is None:
            if jp_fp.embedding and en_fp.embedding and self.embedding_engine:
                emb_sim = self.embedding_engine.similarity(jp_fp.embedding, en_fp.embedding)
            else:
                emb_sim = 0.0
        
        # Title similarity
        title_sim = self._compute_title_similarity(jp_fp.title, en_fp.title)
        
        # Character overlap
        char_overlap = self._compute_character_overlap(jp_fp.characters, en_fp.characters)
        
        # Event overlap
        event_overlap = self._compute_event_overlap(jp_fp.events, en_fp.events)
        
        # Weighted sum
        total = (
            weights.get('embedding', 0.5) * emb_sim +
            weights.get('title', 0.2) * title_sim +
            weights.get('character', 0.2) * char_overlap +
            weights.get('event', 0.1) * event_overlap
        )
        
        return SimilarityScore(
            total=total,
            embedding_similarity=emb_sim,
            title_similarity=title_sim,
            character_overlap=char_overlap,
            event_overlap=event_overlap,
            weights=weights,
        )
    
    def _compute_title_similarity(self, title_jp: Optional[str], title_en: Optional[str]) -> float:
        """
        Compute similarity between titles.
        
        Uses a combination of:
        - String similarity (for direct matches)
        - Length ratio penalty
        - Empty title handling
        """
        if not title_jp or not title_en:
            return 0.0
        
        # Normalize titles
        t1 = title_jp.lower().strip()
        t2 = title_en.lower().strip()
        
        # Exact match after normalization
        if t1 == t2:
            return 1.0
        
        # Check for substring matches
        if t1 in t2 or t2 in t1:
            return 0.8
        
        # Use sequence matching for fuzzy similarity
        ratio = SequenceMatcher(None, t1, t2).ratio()
        
        # Penalize large length differences
        len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2))
        
        return ratio * 0.7 + len_ratio * 0.3
    
    def _compute_character_overlap(
        self,
        chars_jp: List[str],
        chars_en: List[str]
    ) -> float:
        """
        Compute character name overlap score.
        
        Uses Jaccard similarity with fuzzy matching.
        """
        if not chars_jp or not chars_en:
            return 0.0
        
        set_jp = set(c.lower() for c in chars_jp)
        set_en = set(c.lower() for c in chars_en)
        
        # Exact overlap
        intersection = set_jp & set_en
        union = set_jp | set_en
        
        if not union:
            return 0.0
        
        exact_jaccard = len(intersection) / len(union)
        
        # Fuzzy overlap for transliterated names
        fuzzy_matches = 0
        for jp_char in set_jp:
            for en_char in set_en:
                if jp_char not in set_en and en_char not in set_jp:
                    # Check for potential transliteration match
                    if self._is_potential_transliteration(jp_char, en_char):
                        fuzzy_matches += 1
        
        fuzzy_score = fuzzy_matches / max(len(set_jp), len(set_en)) if set_jp or set_en else 0.0
        
        return 0.7 * exact_jaccard + 0.3 * fuzzy_score
    
    def _is_potential_transliteration(self, jp_name: str, en_name: str) -> bool:
        """
        Check if two names might be transliterations of each other.
        
        This is a simplified check; could be enhanced with proper
        romanization tables.
        """
        # Very basic heuristic: similar length and some character overlap
        if abs(len(jp_name) - len(en_name)) > 3:
            return False
        
        # Check if any characters match (for katakana-romaji pairs)
        jp_chars = set(jp_name)
        en_chars = set(en_name.lower())
        
        # Look for common romanization patterns
        common_patterns = [
            ('sh', 'sh'), ('ch', 'ch'), ('tsu', 'tu'), ('ou', 'o'),
        ]
        
        return len(jp_chars & en_chars) >= 2
    
    def _compute_event_overlap(
        self,
        events_jp: List[str],
        events_en: List[str]
    ) -> float:
        """
        Compute event overlap score.
        
        Uses keyword-based matching since exact event matching is difficult.
        """
        if not events_jp or not events_en:
            return 0.0
        
        # Simple approach: count events that have similar descriptions
        matches = 0
        for jp_event in events_jp:
            for en_event in events_en:
                if SequenceMatcher(None, jp_event, en_event).ratio() > 0.5:
                    matches += 1
                    break
        
        return matches / max(len(events_jp), len(events_en))
    
    def get_score(self, jp_idx: int, en_idx: int) -> Optional[SimilarityScore]:
        """Get detailed score for a specific pair."""
        if self._detailed_matrix and jp_idx < len(self._detailed_matrix):
            if en_idx < len(self._detailed_matrix[jp_idx]):
                return self._detailed_matrix[jp_idx][en_idx]
        return None
    
    def get_best_matches_for_jp(self, jp_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get best EN matches for a JP chapter."""
        if self._matrix is None:
            return []
        
        scores = self._matrix[jp_idx]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in indices]
    
    def get_best_matches_for_en(self, en_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get best JP matches for an EN chapter."""
        if self._matrix is None:
            return []
        
        scores = self._matrix[:, en_idx]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in indices]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to dictionary for serialization."""
        return {
            'shape': [self.n_jp, self.n_en],
            'jp_files': [fp.file_path for fp in self.jp_fingerprints],
            'en_files': [fp.file_path for fp in self.en_fingerprints],
            'scores': self._matrix.tolist() if self._matrix is not None else None,
        }


class SimilarityScorer:
    """
    High-level interface for computing similarities.
    
    Integrates fingerprint generation and similarity computation.
    """
    
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the scorer.
        
        Args:
            embedding_engine: Engine for embedding computation
            weights: Similarity component weights
        """
        self.embedding_engine = embedding_engine
        self.weights = weights or {
            'embedding': 0.5,
            'title': 0.2,
            'character': 0.2,
            'event': 0.1,
        }
    
    def compute_matrix(
        self,
        jp_fingerprints: List[ChapterFingerprint],
        en_fingerprints: List[ChapterFingerprint],
    ) -> SimilarityMatrix:
        """
        Compute similarity matrix from fingerprints.
        
        Args:
            jp_fingerprints: Japanese chapter fingerprints
            en_fingerprints: English chapter fingerprints
            
        Returns:
            Computed SimilarityMatrix
        """
        matrix = SimilarityMatrix(
            jp_fingerprints=jp_fingerprints,
            en_fingerprints=en_fingerprints,
            embedding_engine=self.embedding_engine,
        )
        
        matrix.compute(weights=self.weights)
        return matrix
