import torch
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name: str, device: str = "cpu"):
        # Auto-fallback to CPU if CUDA is requested but unavailable
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> List[List[float]]:
        # Vectorized encoding operations
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()