"""
Configuration module for Novel Aligner.

All configurable parameters are centralized here for easy customization.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    # Available models: BAAI/bge-m3, intfloat/multilingual-e5-large, paraphrase-multilingual-mpnet-base-v2
    model_name: str = "BAAI/bge-m3"
    
    # Maximum input length for the model
    max_length: int = 512
    
    # Batch size for embedding generation
    batch_size: int = 32
    
    # Use GPU if available
    use_gpu: bool = True
    
    # Cache directory for embeddings
    cache_dir: str = ".embedding_cache"
    

@dataclass
class FingerprintConfig:
    """Configuration for chapter fingerprint generation."""
    # Number of characters to sample from chapter start
    sample_chars: int = 2000
    
    # Number of characters to sample from chapter end
    sample_end_chars: int = 1000
    
    # Maximum number of characters to extract for title
    max_title_length: int = 100
    
    # Enable summary generation
    use_summary: bool = True
    
    # Summary mode: "raw" or "summary"
    embedding_mode: str = "summary"
    

@dataclass
class SimilarityConfig:
    """Configuration for similarity matrix computation."""
    # Weights for different similarity components
    embedding_weight: float = 0.5
    title_weight: float = 0.2
    character_weight: float = 0.2
    event_weight: float = 0.1
    
    # Minimum confidence threshold for matching
    min_confidence: float = 0.3
    
    # Gap penalty for sequence alignment (insertion/deletion)
    gap_penalty: float = -0.2
    
    # Mismatch penalty
    mismatch_penalty: float = -0.1
    

@dataclass
class AlignmentConfig:
    """Configuration for sequence alignment."""
    # Use global alignment (Needleman-Wunsch) vs local (Smith-Waterman)
    global_alignment: bool = True
    
    # Allow one-to-many mappings (chapter splits)
    allow_splits: bool = False
    
    # Allow many-to-one mappings (chapter merges)
    allow_merges: bool = False
    
    # Maximum skip distance in alignment
    max_skip: int = 5
    

@dataclass
class DatabaseConfig:
    """Configuration for database."""
    # Database file path
    db_path: str = "novel_aligner.db"
    
    # Enable WAL mode for better concurrent reads
    wal_mode: bool = True
    

@dataclass
class APIConfig:
    """Configuration for REST API."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    

@dataclass
class Config:
    """Main configuration container."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if 'embedding' in data:
            config.embedding = EmbeddingConfig(**data['embedding'])
        if 'fingerprint' in data:
            config.fingerprint = FingerprintConfig(**data['fingerprint'])
        if 'similarity' in data:
            config.similarity = SimilarityConfig(**data['similarity'])
        if 'alignment' in data:
            config.alignment = AlignmentConfig(**data['alignment'])
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'api' in data:
            config.api = APIConfig(**data['api'])
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        data = {
            'embedding': self.embedding.__dict__,
            'fingerprint': self.fingerprint.__dict__,
            'similarity': self.similarity.__dict__,
            'alignment': self.alignment.__dict__,
            'database': self.database.__dict__,
            'api': self.api.__dict__,
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


# Global default configuration
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get configuration, checking environment variables first."""
    config = DEFAULT_CONFIG
    
    # Override with environment variables if set
    if os.getenv('EMBEDDING_MODEL'):
        config.embedding.model_name = os.getenv('EMBEDDING_MODEL')
    if os.getenv('DB_PATH'):
        config.database.db_path = os.getenv('DB_PATH')
    
    return config
