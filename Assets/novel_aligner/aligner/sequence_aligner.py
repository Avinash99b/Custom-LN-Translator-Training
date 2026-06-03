import numpy as np
from typing import Tuple, List

class SequenceAligner:
    def __init__(self, skip_penalty: float = -0.15, match_threshold: float = 0.5):
        self.skip_penalty = skip_penalty
        self.match_threshold = match_threshold

    def align(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        n, m = similarity_matrix.shape
        dp = np.zeros((n + 1, m + 1))
        # 0: Diagonal (Match), 1: Up (Skip JP), 2: Left (Skip EN)
        pointers = np.zeros((n + 1, m + 1), dtype=int)

        # Initialize base cases with penalty multipliers
        for i in range(1, n + 1):
            dp[i][0] = dp[i-1][0] + self.skip_penalty
            pointers[i][0] = 1
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j-1] + self.skip_penalty
            pointers[0][j] = 2

        # DP Computation
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score = similarity_matrix[i-1][j-1]
                # Shift score so matches below threshold are penalized
                match_score = dp[i-1][j-1] + (score - self.match_threshold)
                skip_jp = dp[i-1][j] + self.skip_penalty
                skip_en = dp[i][j-1] + self.skip_penalty

                best = max(match_score, skip_jp, skip_en)
                dp[i][j] = best

                if best == match_score:
                    pointers[i][j] = 0
                elif best == skip_jp:
                    pointers[i][j] = 1
                else:
                    pointers[i][j] = 2

        # Backtracking
        i, j = n, m
        alignments = []
        while i > 0 and j > 0:
            if pointers[i][j] == 0:
                score = similarity_matrix[i-1][j-1]
                if score >= self.match_threshold: # Only keep viable matches
                    alignments.append((i - 1, j - 1, float(score)))
                i -= 1
                j -= 1
            elif pointers[i][j] == 1:
                i -= 1
            else:
                j -= 1

        return alignments[::-1] # Reverse to chronological order