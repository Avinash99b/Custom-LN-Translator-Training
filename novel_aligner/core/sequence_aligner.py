"""
Sequence Aligner Module

Implements dynamic programming sequence alignment for chapter matching.
Uses a Needleman-Wunsch style algorithm adapted for chapter alignment.

Key constraints:
- Chapter ordering must be preserved (monotonic alignment)
- Allows insertions, deletions, and skips
- Produces optimal global alignment
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlignmentDirection(Enum):
    """Direction in the DP matrix."""
    DIAGONAL = 0  # Match/mismatch
    UP = 1        # Gap in EN (JP chapter unmatched)
    LEFT = 2      # Gap in JP (EN chapter unmatched)


@dataclass
class AlignmentMatch:
    """
    Represents a single alignment match.
    
    Attributes:
        jp_idx: Index in Japanese chapters (0-based)
        en_idx: Index in English chapters (0-based)
        confidence: Confidence score for this match
        jp_file: File path of Japanese chapter
        en_file: File path of English chapter
    """
    jp_idx: int
    en_idx: int
    confidence: float
    jp_file: str = ""
    en_file: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'jp_idx': self.jp_idx,
            'en_idx': self.en_idx,
            'confidence': self.confidence,
            'jp_file': self.jp_file,
            'en_file': self.en_file,
        }


@dataclass
class AlignmentResult:
    """
    Result of sequence alignment.
    
    Attributes:
        matches: List of matched chapter pairs
        unmatched_jp: Indices of unmatched Japanese chapters
        unmatched_en: Indices of unmatched English chapters
        total_score: Total alignment score
        average_confidence: Average confidence of matches
        alignment_matrix: The DP matrix used (for debugging)
    """
    matches: List[AlignmentMatch]
    unmatched_jp: List[int]
    unmatched_en: List[int]
    total_score: float
    average_confidence: float
    alignment_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'matches': [m.to_dict() for m in self.matches],
            'unmatched_jp': self.unmatched_jp,
            'unmatched_en': self.unmatched_en,
            'total_score': self.total_score,
            'average_confidence': self.average_confidence,
            'num_matches': len(self.matches),
            'num_unmatched_jp': len(self.unmatched_jp),
            'num_unmatched_en': len(self.unmatched_en),
        }


class SequenceAligner:
    """
    Dynamic programming sequence aligner for chapter matching.
    
    This implements a variant of the Needleman-Wunsch algorithm,
    modified for the chapter alignment problem.
    
    Algorithm Explanation:
    ----------------------
    
    Given:
    - JP chapters: J[0..n-1]
    - EN chapters: E[0..m-1]
    - Similarity matrix: S[i][j] = similarity between J[i] and E[j]
    
    We build a DP matrix M of size (n+1) x (m+1) where:
    - M[i][j] = best alignment score for J[0..i-1] and E[0..j-1]
    
    Recurrence relation:
    ```
    M[i][j] = max(
        M[i-1][j-1] + S[i-1][j-1],           # Match J[i-1] with E[j-1]
        M[i-1][j] + gap_penalty,              # Skip J[i-1] (insertion in EN)
        M[i][j-1] + gap_penalty               # Skip E[j-1] (deletion from EN)
    )
    ```
    
    Base cases:
    ```
    M[0][0] = 0
    M[i][0] = i * gap_penalty  (all JP chapters unmatched)
    M[0][j] = j * gap_penalty  (all EN chapters unmatched)
    ```
    
    After filling the matrix, we backtrack from M[n][m] to find the
    optimal alignment path.
    
    Complexity Analysis:
    --------------------
    - Time: O(n * m) for matrix filling + O(n + m) for backtracking
    - Space: O(n * m) for the full matrix
    
    For typical novels with 300-1000 chapters:
    - 500 x 500 = 250,000 cells (very fast)
    - 1000 x 1000 = 1,000,000 cells (still fast)
    
    This is well within the O(n²) requirement and avoids O(n³).
    """
    
    def __init__(
        self,
        gap_penalty: float = -0.2,
        mismatch_penalty: float = -0.1,
        min_confidence: float = 0.3,
        require_monotonic: bool = True,
    ):
        """
        Initialize the aligner.
        
        Args:
            gap_penalty: Penalty for unmatched chapters (insertion/deletion)
            mismatch_penalty: Additional penalty for low-confidence matches
            min_confidence: Minimum confidence to consider a match
            require_monotonic: Enforce monotonic alignment (order preservation)
        """
        self.gap_penalty = gap_penalty
        self.mismatch_penalty = mismatch_penalty
        self.min_confidence = min_confidence
        self.require_monotonic = require_monotonic
        
        # State for current alignment
        self._similarity_matrix: Optional[np.ndarray] = None
        self._dp_matrix: Optional[np.ndarray] = None
        self._traceback_matrix: Optional[np.ndarray] = None
    
    def align(
        self,
        similarity_matrix: np.ndarray,
        jp_files: Optional[List[str]] = None,
        en_files: Optional[List[str]] = None,
    ) -> AlignmentResult:
        """
        Perform sequence alignment.
        
        Args:
            similarity_matrix: Matrix of shape (n_jp, n_en) with similarity scores
            jp_files: Optional list of Japanese file paths
            en_files: Optional list of English file paths
            
        Returns:
            AlignmentResult with matches and unmatched chapters
        """
        self._similarity_matrix = similarity_matrix
        n_jp, n_en = similarity_matrix.shape
        
        logger.info(f"Starting alignment for {n_jp} JP x {n_en} EN chapters")
        
        # Initialize matrices
        self._initialize_matrices(n_jp, n_en)
        
        # Fill DP matrix
        self._fill_dp_matrix(similarity_matrix)
        
        # Backtrack to find alignment
        matches = self._backtrack(jp_files, en_files)
        
        # Find unmatched chapters
        matched_jp = {m.jp_idx for m in matches}
        matched_en = {m.en_idx for m in matches}
        
        unmatched_jp = [i for i in range(n_jp) if i not in matched_jp]
        unmatched_en = [j for j in range(n_en) if j not in matched_en]
        
        # Calculate statistics
        total_score = sum(m.confidence for m in matches)
        avg_confidence = total_score / len(matches) if matches else 0.0
        
        result = AlignmentResult(
            matches=matches,
            unmatched_jp=unmatched_jp,
            unmatched_en=unmatched_en,
            total_score=total_score,
            average_confidence=avg_confidence,
            alignment_matrix=self._dp_matrix.copy(),
        )
        
        logger.info(
            f"Alignment complete: {len(matches)} matches, "
            f"{len(unmatched_jp)} unmatched JP, {len(unmatched_en)} unmatched EN"
        )
        
        return result
    
    def _initialize_matrices(self, n_jp: int, n_en: int) -> None:
        """Initialize DP and traceback matrices."""
        # DP matrix: (n_jp + 1) x (n_en + 1)
        self._dp_matrix = np.zeros((n_jp + 1, n_en + 1), dtype=np.float64)
        
        # Traceback matrix: stores direction for each cell
        self._traceback_matrix = np.zeros((n_jp + 1, n_en + 1), dtype=np.int8)
        
        # Initialize base cases
        for i in range(1, n_jp + 1):
            self._dp_matrix[i, 0] = i * self.gap_penalty
            self._traceback_matrix[i, 0] = AlignmentDirection.UP.value
        
        for j in range(1, n_en + 1):
            self._dp_matrix[0, j] = j * self.gap_penalty
            self._traceback_matrix[0, j] = AlignmentDirection.LEFT.value
    
    def _fill_dp_matrix(self, similarity_matrix: np.ndarray) -> None:
        """
        Fill the DP matrix using the recurrence relation.
        
        Vectorized implementation for performance.
        """
        n_jp, n_en = similarity_matrix.shape
        
        for i in range(1, n_jp + 1):
            for j in range(1, n_en + 1):
                # Get similarity score
                sim = similarity_matrix[i - 1, j - 1]
                
                # Apply mismatch penalty for low-confidence matches
                if sim < self.min_confidence:
                    effective_sim = sim + self.mismatch_penalty
                else:
                    effective_sim = sim
                
                # Calculate three possible scores
                diagonal = self._dp_matrix[i - 1, j - 1] + effective_sim  # Match
                up = self._dp_matrix[i - 1, j] + self.gap_penalty         # Skip JP
                left = self._dp_matrix[i, j - 1] + self.gap_penalty       # Skip EN
                
                # Take maximum and record direction
                scores = [diagonal, up, left]
                best_idx = np.argmax(scores)
                
                self._dp_matrix[i, j] = scores[best_idx]
                self._traceback_matrix[i, j] = best_idx
    
    def _backtrack(
        self,
        jp_files: Optional[List[str]],
        en_files: Optional[List[str]],
    ) -> List[AlignmentMatch]:
        """
        Backtrack through the DP matrix to find the optimal alignment.
        
        Starts from bottom-right and follows the traceback pointers.
        """
        if self._dp_matrix is None or self._traceback_matrix is None:
            raise ValueError("DP matrix not computed")
        
        n_jp = self._dp_matrix.shape[0] - 1
        n_en = self._dp_matrix.shape[1] - 1
        
        matches = []
        i, j = n_jp, n_en
        
        while i > 0 and j > 0:
            direction = self._traceback_matrix[i, j]
            
            if direction == AlignmentDirection.DIAGONAL.value:
                # Match: J[i-1] <-> E[j-1]
                confidence = float(self._similarity_matrix[i - 1, j - 1])
                
                if confidence >= self.min_confidence:
                    match = AlignmentMatch(
                        jp_idx=i - 1,
                        en_idx=j - 1,
                        confidence=confidence,
                        jp_file=jp_files[i - 1] if jp_files else "",
                        en_file=en_files[j - 1] if en_files else "",
                    )
                    matches.append(match)
                
                i -= 1
                j -= 1
                
            elif direction == AlignmentDirection.UP.value:
                # Gap in EN: J[i-1] is unmatched
                i -= 1
                
            elif direction == AlignmentDirection.LEFT.value:
                # Gap in JP: E[j-1] is unmatched
                j -= 1
        
        # Reverse to get matches in order
        matches.reverse()
        
        return matches
    
    def align_with_threshold(
        self,
        similarity_matrix: np.ndarray,
        threshold: float,
        jp_files: Optional[List[str]] = None,
        en_files: Optional[List[str]] = None,
    ) -> AlignmentResult:
        """
        Perform alignment with a custom confidence threshold.
        
        Matches below the threshold are treated as gaps.
        """
        # Create modified similarity matrix
        modified_matrix = similarity_matrix.copy()
        modified_matrix[modified_matrix < threshold] = self.gap_penalty
        
        return self.align(modified_matrix, jp_files, en_files)
    
    def get_alignment_path(self) -> List[Tuple[int, int]]:
        """
        Get the full alignment path including gaps.
        
        Returns list of (jp_idx, en_idx) tuples where -1 indicates a gap.
        """
        if self._dp_matrix is None or self._traceback_matrix is None:
            return []
        
        n_jp = self._dp_matrix.shape[0] - 1
        n_en = self._dp_matrix.shape[1] - 1
        
        path = []
        i, j = n_jp, n_en
        
        while i > 0 or j > 0:
            if i == 0:
                # Only EN chapters left (gaps in JP)
                path.append((-1, j - 1))
                j -= 1
            elif j == 0:
                # Only JP chapters left (gaps in EN)
                path.append((i - 1, -1))
                i -= 1
            else:
                direction = self._traceback_matrix[i, j]
                
                if direction == AlignmentDirection.DIAGONAL.value:
                    path.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif direction == AlignmentDirection.UP.value:
                    path.append((i - 1, -1))
                    i -= 1
                else:
                    path.append((-1, j - 1))
                    j -= 1
        
        path.reverse()
        return path
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute alignment statistics."""
        if self._dp_matrix is None:
            return {}
        
        return {
            'final_score': float(self._dp_matrix[-1, -1]),
            'matrix_shape': list(self._dp_matrix.shape),
            'gap_penalty': self.gap_penalty,
            'min_confidence': self.min_confidence,
        }


def explain_alignment(result: AlignmentResult) -> str:
    """
    Generate a human-readable explanation of the alignment.
    
    Useful for debugging and validation.
    """
    lines = [
        "=" * 60,
        "SEQUENCE ALIGNMENT RESULT",
        "=" * 60,
        "",
        f"Total matches: {len(result.matches)}",
        f"Unmatched JP chapters: {len(result.unmatched_jp)}",
        f"Unmatched EN chapters: {len(result.unmatched_en)}",
        f"Total score: {result.total_score:.3f}",
        f"Average confidence: {result.average_confidence:.3f}",
        "",
        "-" * 60,
        "MATCHES:",
        "-" * 60,
    ]
    
    for i, match in enumerate(result.matches):
        lines.append(
            f"  {i + 1}. JP[{match.jp_idx}] ({match.jp_file}) "
            f"<-> EN[{match.en_idx}] ({match.en_file}) "
            f"[confidence: {match.confidence:.3f}]"
        )
    
    if result.unmatched_jp:
        lines.extend([
            "",
            "-" * 60,
            "UNMATCHED JP CHAPTERS:",
            "-" * 60,
        ])
        for idx in result.unmatched_jp:
            lines.append(f"  - JP[{idx}]")
    
    if result.unmatched_en:
        lines.extend([
            "",
            "-" * 60,
            "UNMATCHED EN CHAPTERS:",
            "-" * 60,
        ])
        for idx in result.unmatched_en:
            lines.append(f"  - EN[{idx}]")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)
