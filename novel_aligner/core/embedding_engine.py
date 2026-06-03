"""
Embedding Engine Module

Provides multilingual embedding generation with support for:
- BAAI/bge-m3
- intfloat/multilingual-e5-large
- paraphrase-multilingual-mpnet-base-v2

Implements caching to avoid recomputation.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    model_name: str
    dimension: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'embedding': self.embedding,
            'model_name': self.model_name,
            'dimension': self.dimension,
        }


class EmbeddingBackend:
    """
    Abstract base class for embedding backends.
    """
    
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def dimension(self) -> int:
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Backend using sentence-transformers library.
    """
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        use_gpu: bool = True
    ):
        super().__init__(model_name, max_length)
        
        try:
            from sentence_transformers import SentenceTransformer
            
            device = 'cuda' if use_gpu and self._check_gpu() else 'cpu'
            self.model = SentenceTransformer(model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Loaded {model_name} on {device}")
        except ImportError:
            raise ImportError("sentence-transformers not installed")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


class TransformersBackend(EmbeddingBackend):
    """
    Backend using transformers library directly.
    Used for models not supported by sentence-transformers.
    """
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        use_gpu: bool = True
    ):
        super().__init__(model_name, max_length)
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            if use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            
            self.model.eval()
            
            # Get dimension by running a test
            test_input = self.tokenizer(["test"], return_tensors="pt", truncation=True, max_length=max_length)
            if self.device == 'cuda':
                test_input = {k: v.cuda() for k, v in test_input.items()}
            
            with torch.no_grad():
                output = self.model(**test_input)
                self._dimension = output.last_hidden_state.shape[-1]
            
            logger.info(f"Loaded {model_name} on {self.device}")
            
        except ImportError:
            raise ImportError("transformers not installed")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Encode texts to embeddings using mean pooling."""
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                # Normalize
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / (norms + 1e-9)
            
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingEngine:
    """
    Main embedding engine with caching support.
    
    Supports multiple backend models and provides automatic caching
    to avoid recomputing embeddings.
    """
    
    # Supported models and their preferred backends
    SUPPORTED_MODELS = {
        'BAAI/bge-m3': 'sentence_transformer',
        'intfloat/multilingual-e5-large': 'sentence_transformer',
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': 'sentence_transformer',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'sentence_transformer',
    }
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str = ".embedding_cache",
        use_gpu: bool = True,
        max_length: int = 512,
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the embedding model
            cache_dir: Directory for caching embeddings
            use_gpu: Whether to use GPU if available
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.use_gpu = use_gpu
        self.max_length = max_length
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend
        self.backend = self._create_backend(model_name, use_gpu, max_length)
        
        logger.info(f"Initialized EmbeddingEngine with {model_name}")
    
    def _create_backend(
        self,
        model_name: str,
        use_gpu: bool,
        max_length: int
    ) -> EmbeddingBackend:
        """Create appropriate backend for the model."""
        backend_type = self.SUPPORTED_MODELS.get(model_name, 'sentence_transformer')
        
        if backend_type == 'sentence_transformer':
            return SentenceTransformerBackend(model_name, max_length, use_gpu)
        elif backend_type == 'transformers':
            return TransformersBackend(model_name, max_length, use_gpu)
        else:
            # Default to sentence transformer
            return SentenceTransformerBackend(model_name, max_length, use_gpu)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a text."""
        return hashlib.sha256(f"{self.model_name}:{text}".encode('utf-8')).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache if available."""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('model') == self.model_name:
                        return data.get('embedding')
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            data = {
                'model': self.model_name,
                'embedding': embedding,
                'dimension': len(embedding),
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            use_cache: Whether to use caching
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Single embedding or list of embeddings
        """
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        results = []
        
        # Process with caching
        for i, text in enumerate(texts):
            if use_cache:
                cache_key = self._get_cache_key(text)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    results.append(cached)
                    continue
            
            # Need to compute
            results.append(None)
        
        # Find texts that need computation
        texts_to_compute = []
        indices_to_compute = []
        
        for i, (text, result) in enumerate(zip(texts, results)):
            if result is None:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        # Compute embeddings for remaining texts
        if texts_to_compute:
            embeddings = self.backend.encode(
                texts_to_compute,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            
            # Store results
            for idx, emb_idx in zip(indices_to_compute, range(len(texts_to_compute))):
                embedding = embeddings[emb_idx].tolist()
                results[idx] = embedding
                
                # Cache if enabled
                if use_cache:
                    cache_key = self._get_cache_key(texts_to_compute[emb_idx])
                    self._save_to_cache(cache_key, embedding)
        
        if single_text:
            return results[0]
        return results
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Encode a batch of texts efficiently.
        
        For large batches, bypasses individual caching overhead.
        """
        embeddings = self.backend.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.backend.dimension
    
    def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
            
        Returns:
            Similarity score
        """
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        
        if metric == 'cosine':
            # For normalized embeddings, dot product equals cosine similarity
            return float(np.dot(e1, e2))
        elif metric == 'dot':
            return float(np.dot(e1, e2))
        elif metric == 'euclidean':
            return float(-np.linalg.norm(e1 - e2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def similarity_matrix(
        self,
        embeddings1: List[List[float]],
        embeddings2: List[List[float]],
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (n x d)
            embeddings2: Second set of embeddings (m x d)
            metric: Similarity metric
            
        Returns:
            Similarity matrix (n x m)
        """
        e1 = np.array(embeddings1)
        e2 = np.array(embeddings2)
        
        if metric == 'cosine':
            # For normalized embeddings, matrix multiplication gives cosine similarities
            return np.dot(e1, e2.T)
        elif metric == 'dot':
            return np.dot(e1, e2.T)
        elif metric == 'euclidean':
            # Compute pairwise Euclidean distances
            diff = e1[:, np.newaxis, :] - e2[np.newaxis, :, :]
            return -np.linalg.norm(diff, axis=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        cache_files = list(self.cache_dir.glob("**/*.json"))
        cache_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'use_gpu': self.use_gpu,
            'max_length': self.max_length,
            'cache_entries': len(cache_files),
            'cache_size_bytes': cache_size,
            'cache_dir': str(self.cache_dir),
        }
