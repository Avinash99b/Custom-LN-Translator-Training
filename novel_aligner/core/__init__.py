"""
Core module for Novel Aligner.

Contains the main processing components for chapter alignment.
"""
from .file_loader import FileLoader, ChapterData
from .chapter_fingerprint import ChapterFingerprint, FingerprintGenerator
from .embedding_engine import EmbeddingEngine, EmbeddingBackend
from .similarity_matrix import SimilarityMatrix, SimilarityScorer
from .sequence_aligner import SequenceAligner, AlignmentResult

__all__ = [
    'FileLoader',
    'ChapterData',
    'ChapterFingerprint',
    'FingerprintGenerator',
    'EmbeddingEngine',
    'EmbeddingBackend',
    'SimilarityMatrix',
    'SimilarityScorer',
    'SequenceAligner',
    'AlignmentResult',
]
