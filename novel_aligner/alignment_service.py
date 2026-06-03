"""
Main Alignment Service

Orchestrates the complete alignment pipeline from file loading to graph generation.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .config import Config, get_config
from .core.file_loader import FileLoader, ChapterData
from .core.chapter_fingerprint import FingerprintGenerator, ChapterFingerprint
from .core.embedding_engine import EmbeddingEngine
from .core.similarity_matrix import SimilarityMatrix, SimilarityScorer
from .core.sequence_aligner import SequenceAligner, AlignmentResult
from .graph_builder import GraphBuilder
from .database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class AlignmentService:
    """
    Main service for orchestrating chapter alignment.
    
    Coordinates all components of the alignment pipeline.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        """
        Initialize the alignment service.
        
        Args:
            config: Configuration object
            db_manager: Database manager instance
        """
        self.config = config or get_config()
        self.db_manager = db_manager
        
        # Components (initialized per-alignment)
        self.embedding_engine: Optional[EmbeddingEngine] = None
        self.fingerprint_generator: Optional[FingerprintGenerator] = None
    
    def align_novel(
        self,
        novel_root: str,
        rebuild_embeddings: bool = False,
        embedding_model: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Perform complete alignment for a novel.
        
        Args:
            novel_root: Path to novel root directory
            rebuild_embeddings: Whether to regenerate embeddings
            embedding_model: Override embedding model
            min_confidence: Minimum confidence threshold
            
        Returns:
            Alignment result dictionary
        """
        logger.info(f"Starting alignment for {novel_root}")
        
        # Step 1: Load chapters
        logger.info("Step 1: Loading chapters...")
        loader = FileLoader(novel_root)
        jp_chapters, en_chapters = loader.load_all()
        
        if not jp_chapters or not en_chapters:
            raise ValueError(f"No chapters found in {novel_root}")
        
        stats = loader.get_statistics()
        logger.info(f"Loaded {len(jp_chapters)} JP and {len(en_chapters)} EN chapters")
        
        # Step 2: Register novel in database
        novel_id = None
        if self.db_manager:
            novel_id = self.db_manager.create_novel(
                name=loader.get_novel_name(),
                root_path=novel_root,
                jp_count=len(jp_chapters),
                en_count=len(en_chapters),
            )
            logger.info(f"Novel registered with ID: {novel_id}")
        
        # Step 3: Generate fingerprints
        logger.info("Step 2: Generating fingerprints...")
        self.fingerprint_generator = FingerprintGenerator(
            sample_chars=self.config.fingerprint.sample_chars,
            sample_end_chars=self.config.fingerprint.sample_end_chars,
            use_summary=self.config.fingerprint.use_summary,
            embedding_mode=self.config.fingerprint.embedding_mode,
        )
        
        jp_fingerprints = self._generate_fingerprints(jp_chapters, 'jp')
        en_fingerprints = self._generate_fingerprints(en_chapters, 'en')
        
        # Step 4: Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        model_name = embedding_model or self.config.embedding.model_name
        self.embedding_engine = EmbeddingEngine(
            model_name=model_name,
            cache_dir=self.config.embedding.cache_dir,
            use_gpu=self.config.embedding.use_gpu,
            max_length=self.config.embedding.max_length,
        )
        
        jp_fingerprints = self._compute_embeddings(
            jp_fingerprints, rebuild_embeddings, model_name
        )
        en_fingerprints = self._compute_embeddings(
            en_fingerprints, rebuild_embeddings, model_name
        )
        
        # Save fingerprints and embeddings to database
        if self.db_manager and novel_id:
            self._save_to_database(
                novel_id, jp_chapters, en_chapters,
                jp_fingerprints, en_fingerprints
            )
        
        # Step 5: Compute similarity matrix
        logger.info("Step 4: Computing similarity matrix...")
        scorer = SimilarityScorer(
            embedding_engine=self.embedding_engine,
            weights={
                'embedding': self.config.similarity.embedding_weight,
                'title': self.config.similarity.title_weight,
                'character': self.config.similarity.character_weight,
                'event': self.config.similarity.event_weight,
            },
        )
        
        similarity_matrix = scorer.compute_matrix(jp_fingerprints, en_fingerprints)
        
        # Step 6: Perform sequence alignment
        logger.info("Step 5: Performing sequence alignment...")
        aligner = SequenceAligner(
            gap_penalty=self.config.similarity.gap_penalty,
            mismatch_penalty=self.config.similarity.mismatch_penalty,
            min_confidence=min_confidence,
        )
        
        jp_files = [c.relative_path for c in jp_chapters]
        en_files = [c.relative_path for c in en_chapters]
        
        alignment_result = aligner.align(similarity_matrix, jp_files, en_files)
        
        # Step 7: Build output graph
        logger.info("Step 6: Building output graph...")
        builder = GraphBuilder(loader.get_novel_name())
        
        graph = builder.build(
            alignment_result=alignment_result,
            jp_chapters=jp_chapters,
            en_chapters=en_chapters,
            jp_fingerprints=jp_fingerprints,
            en_fingerprints=en_fingerprints,
        )
        
        # Save relations to database
        if self.db_manager and novel_id:
            self._save_relations(novel_id, alignment_result, jp_chapters, en_chapters)
            
            # Save validation result
            validation_report = builder.generate_validation_report(
                alignment_result, jp_chapters, en_chapters
            )
            self.db_manager.save_validation_result(
                novel_id=novel_id,
                matched=len(alignment_result.matches),
                unmatched_jp=len(alignment_result.unmatched_jp),
                unmatched_en=len(alignment_result.unmatched_en),
                avg_confidence=alignment_result.average_confidence,
                details=validation_report,
            )
        
        logger.info(
            f"Alignment complete: {len(alignment_result.matches)} matches, "
            f"avg confidence: {alignment_result.average_confidence:.3f}"
        )
        
        return {
            'novel': loader.get_novel_name(),
            'novel_id': novel_id,
            'graph': graph,
            'statistics': {
                'total_jp': len(jp_chapters),
                'total_en': len(en_chapters),
                'matched': len(alignment_result.matches),
                'unmatched_jp': len(alignment_result.unmatched_jp),
                'unmatched_en': len(alignment_result.unmatched_en),
                'average_confidence': round(alignment_result.average_confidence, 4),
            },
        }
    
    def _generate_fingerprints(
        self,
        chapters: List[ChapterData],
        language: str,
    ) -> List[ChapterFingerprint]:
        """Generate fingerprints for a list of chapters."""
        fingerprints = []
        
        for chapter in chapters:
            fp = self.fingerprint_generator.generate(
                content=chapter.content,
                title=chapter.title,
                language=language,
                file_path=chapter.relative_path,
            )
            fingerprints.append(fp)
        
        return fingerprints
    
    def _compute_embeddings(
        self,
        fingerprints: List[ChapterFingerprint],
        rebuild: bool,
        model_name: str,
    ) -> List[ChapterFingerprint]:
        """Compute embeddings for fingerprints."""
        texts = []
        indices = []
        
        for i, fp in enumerate(fingerprints):
            # Prepare text for embedding
            if fp.embedding_mode == "summary" and fp.summary:
                text = f"Title: {fp.title or ''}\nSummary: {fp.summary}"
            else:
                text = f"Title: {fp.title or ''}\nContent: {fp.summary or ''}"
            
            texts.append(text)
            indices.append(i)
        
        # Compute embeddings in batch
        embeddings = self.embedding_engine.encode_batch(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress=True,
        )
        
        # Assign embeddings to fingerprints
        for i, emb in zip(indices, embeddings):
            fingerprints[i].embedding = emb
        
        return fingerprints
    
    def _save_to_database(
        self,
        novel_id: int,
        jp_chapters: List[ChapterData],
        en_chapters: List[ChapterData],
        jp_fingerprints: List[ChapterFingerprint],
        en_fingerprints: List[ChapterFingerprint],
    ) -> None:
        """Save chapters and fingerprints to database."""
        # Save JP chapters
        for chapter, fp in zip(jp_chapters, jp_fingerprints):
            chapter_id = self.db_manager.create_chapter(
                novel_id=novel_id,
                file_path=chapter.file_path,
                relative_path=chapter.relative_path,
                language='jp',
                chapter_number=chapter.chapter_number,
                title=chapter.title,
                content_hash=fp.hash,
                content_preview=chapter.content_preview,
                fingerprint_json=fp.to_dict(),
            )
            
            # Save embedding
            if fp.embedding:
                self.db_manager.save_embedding(
                    chapter_id=chapter_id,
                    model_name=self.embedding_engine.model_name,
                    embedding_mode=fp.embedding_mode,
                    vector=fp.embedding,
                    dimension=self.embedding_engine.dimension,
                )
        
        # Save EN chapters
        for chapter, fp in zip(en_chapters, en_fingerprints):
            chapter_id = self.db_manager.create_chapter(
                novel_id=novel_id,
                file_path=chapter.file_path,
                relative_path=chapter.relative_path,
                language='en',
                chapter_number=chapter.chapter_number,
                title=chapter.title,
                content_hash=fp.hash,
                content_preview=chapter.content_preview,
                fingerprint_json=fp.to_dict(),
            )
            
            # Save embedding
            if fp.embedding:
                self.db_manager.save_embedding(
                    chapter_id=chapter_id,
                    model_name=self.embedding_engine.model_name,
                    embedding_mode=fp.embedding_mode,
                    vector=fp.embedding,
                    dimension=self.embedding_engine.dimension,
                )
    
    def _save_relations(
        self,
        novel_id: int,
        result: AlignmentResult,
        jp_chapters: List[ChapterData],
        en_chapters: List[ChapterData],
    ) -> None:
        """Save alignment relations to database."""
        # Get chapter IDs from database
        all_chapters = self.db_manager.get_chapters_by_novel(novel_id)
        jp_by_path = {c['relative_path']: c['id'] for c in all_chapters if c['language'] == 'jp'}
        en_by_path = {c['relative_path']: c['id'] for c in all_chapters if c['language'] == 'en'}
        
        relations = []
        for match in result.matches:
            jp_path = jp_chapters[match.jp_idx].relative_path
            en_path = en_chapters[match.en_idx].relative_path
            
            jp_id = jp_by_path.get(jp_path)
            en_id = en_by_path.get(en_path)
            
            if jp_id and en_id:
                relations.append((jp_id, en_id, match.confidence))
        
        if relations:
            self.db_manager.save_relations_batch(novel_id, relations)
