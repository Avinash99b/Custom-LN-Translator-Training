import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class SimilarityMatrix:
    def __init__(self, weights: Dict[str, float]):
        self.w_e = weights.get('embedding', 0.5)
        self.w_t = weights.get('title', 0.2)
        self.w_c = weights.get('character', 0.2)
        self.w_e_evt = weights.get('event', 0.1)

    def build_matrix(self, jp_fps: List[Dict], en_fps: List[Dict]) -> np.ndarray:
        n_jp, n_en = len(jp_fps), len(en_fps)
        if n_jp == 0 or n_en == 0: return np.zeros((0, 0))

        # 1. Compute vectorized Embedding Similarity (O(N*M))
        jp_emb = np.array([f["embedding"] for f in jp_fps])
        en_emb = np.array([f["embedding"] for f in en_fps])
        sim_matrix = cosine_similarity(jp_emb, en_emb) * self.w_e

        # 2. Add structural modifiers
        for i, jp in enumerate(jp_fps):
            jp_chars = set(jp["characters"])
            jp_events = set(jp["events"])
            for j, en in enumerate(en_fps):
                en_chars = set(en["characters"])
                en_events = set(en["events"])
                
                # Jaccard similarities
                char_sim = len(jp_chars & en_chars) / max(1, len(jp_chars | en_chars))
                evt_sim = len(jp_events & en_events) / max(1, len(jp_events | en_events))
                
                sim_matrix[i, j] += (char_sim * self.w_c) + (evt_sim * self.w_e_evt)
                
                # Title similarity fallback (can use embedding sim of titles if available)
                if jp["title"] and jp["title"] == en["title"]:
                    sim_matrix[i, j] += self.w_t
                    
        return np.clip(sim_matrix, 0.0, 1.0)