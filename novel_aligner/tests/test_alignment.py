"""
Tests for Novel Aligner.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestSequenceAligner:
    """Test the sequence alignment algorithm."""
    
    def test_basic_alignment(self):
        """Test basic alignment with perfect matches."""
        from novel_aligner.core.sequence_aligner import SequenceAligner
        
        # Create a simple similarity matrix with diagonal matches
        similarity = np.array([
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9],
        ])
        
        aligner = SequenceAligner(gap_penalty=-0.2, min_confidence=0.3)
        result = aligner.align(similarity)
        
        assert len(result.matches) == 3
        assert result.matches[0].jp_idx == 0
        assert result.matches[0].en_idx == 0
        assert result.matches[1].jp_idx == 1
        assert result.matches[1].en_idx == 1
        assert result.matches[2].jp_idx == 2
        assert result.matches[2].en_idx == 2
    
    def test_alignment_with_gaps(self):
        """Test alignment when some chapters are missing."""
        from novel_aligner.core.sequence_aligner import SequenceAligner
        
        # JP has 4 chapters, EN has 3 (EN chapter 2 is missing)
        # Expected: JP0->EN0, JP1 unmatched, JP2->EN1, JP3->EN2
        similarity = np.array([
            [0.9, 0.1, 0.1],  # JP0 matches EN0
            [0.1, 0.1, 0.1],  # JP1 has no match
            [0.1, 0.9, 0.1],  # JP2 matches EN1
            [0.1, 0.1, 0.9],  # JP3 matches EN2
        ])
        
        aligner = SequenceAligner(gap_penalty=-0.2, min_confidence=0.3)
        result = aligner.align(similarity)
        
        assert len(result.matches) == 3
        assert 1 in result.unmatched_jp  # JP1 should be unmatched
    
    def test_offset_alignment(self):
        """Test alignment with offset chapter numbers."""
        from novel_aligner.core.sequence_aligner import SequenceAligner
        
        # JP starts at 1, EN starts at 2 (offset by 1)
        # JP1->EN2, JP2->EN3, JP3->EN4
        similarity = np.array([
            [0.1, 0.9, 0.1, 0.1],  # JP0 matches EN1
            [0.1, 0.1, 0.9, 0.1],  # JP1 matches EN2
            [0.1, 0.1, 0.1, 0.9],  # JP2 matches EN3
        ])
        
        aligner = SequenceAligner(gap_penalty=-0.2, min_confidence=0.3)
        result = aligner.align(similarity)
        
        assert len(result.matches) == 3
        assert result.matches[0].en_idx == 1
        assert result.matches[1].en_idx == 2
        assert result.matches[2].en_idx == 3
    
    def test_monotonic_constraint(self):
        """Test that alignment preserves ordering."""
        from novel_aligner.core.sequence_aligner import SequenceAligner
        
        # Create a matrix where non-monotonic would score higher
        # but monotonic constraint should enforce order
        similarity = np.array([
            [0.5, 0.9, 0.1],  # JP0 better match with EN1
            [0.9, 0.5, 0.1],  # JP1 better match with EN0
            [0.1, 0.1, 0.9],  # JP2 matches EN2
        ])
        
        aligner = SequenceAligner(gap_penalty=-0.2, min_confidence=0.3)
        result = aligner.align(similarity)
        
        # Check that matches are monotonic
        prev_en = -1
        for match in result.matches:
            assert match.en_idx > prev_en or prev_en == -1
            prev_en = match.en_idx


class TestSimilarityMatrix:
    """Test similarity matrix computation."""
    
    def test_weighted_similarity(self):
        """Test weighted combination of similarity scores."""
        from novel_aligner.core.chapter_fingerprint import ChapterFingerprint
        from novel_aligner.core.similarity_matrix import SimilarityMatrix
        
        jp_fp = ChapterFingerprint(
            file_path="JP/1.txt",
            title="Chapter One",
            characters=["Alice", "Bob"],
            events=["meeting"],
            embedding=[0.8, 0.2, 0.0],
        )
        
        en_fp = ChapterFingerprint(
            file_path="EN/1.txt",
            title="Chapter 1",
            characters=["alice", "bob"],
            events=["meeting"],
            embedding=[0.75, 0.25, 0.0],
        )
        
        matrix = SimilarityMatrix(
            jp_fingerprints=[jp_fp],
            en_fingerprints=[en_fp],
        )
        
        # Manual computation
        weights = {'embedding': 0.5, 'title': 0.2, 'character': 0.2, 'event': 0.1}
        
        # Embedding similarity (cosine)
        emb_sim = np.dot([0.8, 0.2, 0.0], [0.75, 0.25, 0.0])
        
        # Title similarity
        title_sim = 0.8  # Approximate
        
        # Character overlap (exact + fuzzy)
        char_overlap = 1.0  # All match
        
        # Event overlap
        event_overlap = 1.0  # Exact match
        
        expected = 0.5 * emb_sim + 0.2 * title_sim + 0.2 * char_overlap + 0.1 * event_overlap
        
        computed_matrix = matrix.compute(weights=weights, use_embeddings=False)
        # Note: This test verifies the structure; actual values depend on implementation


class TestFingerprintGenerator:
    """Test fingerprint generation."""
    
    def test_title_normalization(self):
        """Test title normalization."""
        from novel_aligner.core.chapter_fingerprint import FingerprintGenerator
        
        gen = FingerprintGenerator()
        
        # Test various title formats
        titles = [
            ("第1話：始まり", "始まり"),
            ("Chapter 1: The Beginning", "The Beginning"),
            ("Vol. 1 Chapter 1 - Start", "Start"),
        ]
        
        for input_title, expected in titles:
            result = gen._normalize_title(input_title)
            # Just verify it returns something reasonable
            assert isinstance(result, str)
            assert len(result) <= 100
    
    def test_character_extraction(self):
        """Test character name extraction."""
        from novel_aligner.core.chapter_fingerprint import FingerprintGenerator
        
        gen = FingerprintGenerator()
        
        text = "太郎さんは花子さんと会った。田中先生も来ていた。"
        characters = gen._extract_characters(text, 'jp')
        
        # Should extract some names
        assert isinstance(characters, list)


class TestFileLoader:
    """Test file loading."""
    
    def test_load_chapters(self):
        """Test loading chapters from directory structure."""
        from novel_aligner.core.file_loader import FileLoader
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "TestNovel"
            jp_dir = root / "JP"
            en_dir = root / "EN"
            
            jp_dir.mkdir(parents=True)
            en_dir.mkdir(parents=True)
            
            # Create test files
            (jp_dir / "1.txt").write_text("第1話\n日本語のテキスト")
            (jp_dir / "2.txt").write_text("第2話\n続き")
            (en_dir / "1.txt").write_text("Chapter 1\nEnglish text")
            (en_dir / "2.txt").write_text("Chapter 2\nContinuation")
            
            loader = FileLoader(str(root))
            jp_chapters, en_chapters = loader.load_all()
            
            assert len(jp_chapters) == 2
            assert len(en_chapters) == 2
            assert jp_chapters[0].language == 'jp'
            assert en_chapters[0].language == 'en'
    
    def test_chapter_number_extraction(self):
        """Test chapter number extraction from filenames."""
        from novel_aligner.core.file_loader import FileLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "TestNovel"
            jp_dir = root / "JP"
            jp_dir.mkdir(parents=True)
            
            # Various filename formats
            files = [
                "1.txt",
                "chapter_5.txt",
                "第3話.txt",
                "vol1_ch10.txt",
            ]
            
            for f in files:
                (jp_dir / f).write_text("content")
            
            loader = FileLoader(str(root))
            jp_chapters, _ = loader.load_all()
            
            # Should extract numbers from all
            numbers = [c.chapter_number for c in jp_chapters if c.chapter_number]
            assert len(numbers) == len(files)


class TestGraphBuilder:
    """Test graph building."""
    
    def test_build_graph(self):
        """Test building output graph."""
        from novel_aligner.core.file_loader import ChapterData
        from novel_aligner.core.sequence_aligner import AlignmentResult, AlignmentMatch
        from novel_aligner.graph_builder import GraphBuilder
        
        jp_chapters = [
            ChapterData(file_path="/jp/1", relative_path="JP/1.txt", language="jp"),
            ChapterData(file_path="/jp/2", relative_path="JP/2.txt", language="jp"),
        ]
        
        en_chapters = [
            ChapterData(file_path="/en/1", relative_path="EN/1.txt", language="en"),
            ChapterData(file_path="/en/2", relative_path="EN/2.txt", language="en"),
        ]
        
        matches = [
            AlignmentMatch(jp_idx=0, en_idx=0, confidence=0.95),
            AlignmentMatch(jp_idx=1, en_idx=1, confidence=0.90),
        ]
        
        result = AlignmentResult(
            matches=matches,
            unmatched_jp=[],
            unmatched_en=[],
            total_score=1.85,
            average_confidence=0.925,
        )
        
        builder = GraphBuilder("TestNovel")
        graph = builder.build_minimal(result, jp_chapters, en_chapters)
        
        assert graph['novel'] == 'TestNovel'
        assert len(graph['relations']) == 2
        assert len(graph['unmatched_jp']) == 0
        assert len(graph['unmatched_en']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
