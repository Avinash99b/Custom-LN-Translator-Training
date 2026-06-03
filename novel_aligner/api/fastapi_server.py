"""
API Module for Novel Aligner.

FastAPI-based REST API for chapter alignment operations.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


# Pydantic models for request/response

class NovelSummary(BaseModel):
    """Summary of a novel."""
    id: int
    name: str
    root_path: str
    jp_chapter_count: int
    en_chapter_count: int
    
    class Config:
        from_attributes = True


class RelationResponse(BaseModel):
    """Chapter relation response."""
    jp: str
    en: str
    confidence: float
    jp_title: Optional[str] = None
    en_title: Optional[str] = None


class AlignmentRequest(BaseModel):
    """Request to trigger alignment."""
    rebuild_embeddings: bool = False
    embedding_model: Optional[str] = None
    min_confidence: float = 0.3


class AlignmentResponse(BaseModel):
    """Alignment result response."""
    novel: str
    matched: int
    unmatched_jp: int
    unmatched_en: int
    average_confidence: float
    relations: List[RelationResponse]


class ChapterResponse(BaseModel):
    """Chapter details response."""
    id: int
    file_path: str
    relative_path: str
    language: str
    chapter_number: Optional[int] = None
    title: Optional[str] = None
    content_preview: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database_connected: bool


# FastAPI application

app = FastAPI(
    title="Novel Aligner API",
    description="API for automatic Japanese/English light novel chapter alignment",
    version="1.0.0",
)

# Global references (will be set by main.py)
_db_manager = None
_aligner_service = None


def set_dependencies(db_manager, aligner_service):
    """Set global dependencies."""
    global _db_manager, _aligner_service
    _db_manager = db_manager
    _aligner_service = aligner_service


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    db_connected = False
    if _db_manager:
        try:
            _db_manager.list_novels()
            db_connected = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        version="1.0.0",
        database_connected=db_connected,
    )


@app.get("/novels", response_model=List[NovelSummary])
async def list_novels():
    """List all novels in the database."""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    novels = _db_manager.list_novels()
    return [NovelSummary(**n) for n in novels]


@app.get("/novel/{novel_id}", response_model=Dict[str, Any])
async def get_novel(novel_id: int):
    """Get details of a specific novel."""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    novel = _db_manager.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    
    # Get statistics
    stats = _db_manager.get_statistics(novel_id)
    
    return {
        **novel,
        'statistics': stats,
    }


@app.get("/novel/{novel_id}/relations", response_model=List[RelationResponse])
async def get_novel_relations(novel_id: int):
    """Get all chapter relations for a novel."""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    novel = _db_manager.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    
    relations = _db_manager.get_relations_by_novel(novel_id)
    
    return [
        RelationResponse(
            jp=r['jp_path'],
            en=r['en_path'],
            confidence=r['confidence'],
            jp_title=r.get('jp_title'),
            en_title=r.get('en_title'),
        )
        for r in relations
    ]


@app.get("/novel/{novel_id}/unmatched")
async def get_unmatched_chapters(novel_id: int):
    """Get unmatched chapters for a novel."""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    novel = _db_manager.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    
    # Get all chapters
    all_chapters = _db_manager.get_chapters_by_novel(novel_id)
    jp_chapters = [c for c in all_chapters if c['language'] == 'jp']
    en_chapters = [c for c in all_chapters if c['language'] == 'en']
    
    # Get relations
    relations = _db_manager.get_relations_by_novel(novel_id)
    
    matched_jp = {r['jp_path'] for r in relations}
    matched_en = {r['en_path'] for r in relations}
    
    return {
        'novel_id': novel_id,
        'unmatched_jp': [
            {'relative_path': c['relative_path'], 'title': c.get('title')}
            for c in jp_chapters if c['relative_path'] not in matched_jp
        ],
        'unmatched_en': [
            {'relative_path': c['relative_path'], 'title': c.get('title')}
            for c in en_chapters if c['relative_path'] not in matched_en
        ],
    }


@app.post("/novel/{novel_id}/align", response_model=Dict[str, Any])
async def align_novel(
    novel_id: int,
    request: AlignmentRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger alignment for a novel.
    
    This is a long-running operation that can be executed in the background.
    """
    if not _db_manager or not _aligner_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    novel = _db_manager.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    
    try:
        # Run alignment
        result = await asyncio.to_thread(
            _aligner_service.align_novel,
            novel_root=novel['root_path'],
            rebuild_embeddings=request.rebuild_embeddings,
            embedding_model=request.embedding_model,
            min_confidence=request.min_confidence,
        )
        
        return {
            'status': 'completed',
            'novel_id': novel_id,
            'result': result,
        }
    
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chapter/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(chapter_id: int):
    """Get details of a specific chapter."""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    chapter = _db_manager.get_chapter(chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    return ChapterResponse(**chapter)


@app.get("/novel/{novel_id}/graph")
async def export_graph(novel_id: int):
    """Export the complete alignment graph for a novel."""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    novel = _db_manager.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    
    # Get relations
    relations = _db_manager.get_relations_by_novel(novel_id)
    
    # Get all chapters
    all_chapters = _db_manager.get_chapters_by_novel(novel_id)
    jp_chapters = [c for c in all_chapters if c['language'] == 'jp']
    en_chapters = [c for c in all_chapters if c['language'] == 'en']
    
    matched_jp = {r['jp_path'] for r in relations}
    matched_en = {r['en_path'] for r in relations}
    
    return {
        'novel': novel['name'],
        'relations': [
            {
                'jp': r['jp_path'],
                'en': r['en_path'],
                'confidence': r['confidence'],
            }
            for r in relations
        ],
        'unmatched_jp': [
            c['relative_path'] for c in jp_chapters 
            if c['relative_path'] not in matched_jp
        ],
        'unmatched_en': [
            c['relative_path'] for c in en_chapters 
            if c['relative_path'] not in matched_en
        ],
        'metadata': {
            'total_jp': len(jp_chapters),
            'total_en': len(en_chapters),
            'matched': len(relations),
        },
    }


@app.get("/novel/{novel_id}/validation")
async def get_validation(novel_id: int):
    """Get validation results for a novel."""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    novel = _db_manager.get_novel(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")
    
    validation = _db_manager.get_latest_validation(novel_id)
    if not validation:
        return {'status': 'no_validation', 'novel_id': novel_id}
    
    return {
        'status': 'validated',
        'novel_id': novel_id,
        'timestamp': validation.get('run_timestamp'),
        'matched': validation.get('matched_count'),
        'unmatched_jp': validation.get('unmatched_jp_count'),
        'unmatched_en': validation.get('unmatched_en_count'),
        'average_confidence': validation.get('average_confidence'),
        'details': validation.get('details_json'),
    }


# Error handlers

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={'detail': 'Internal server error'},
    )
