#!/usr/bin/env python3
"""
Command Line Interface for Novel Aligner.

Usage:
    python cli.py align /path/to/NovelName
    python cli.py rebuild-embeddings /path/to/NovelName
    python cli.py export-graph /path/to/NovelName output.json
    python cli.py validate /path/to/NovelName
    python cli.py stats /path/to/NovelName
    python cli.py server  # Start API server
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def cmd_align(args):
    """Align chapters in a novel."""
    from .config import Config
    from .database.db_manager import DatabaseManager
    from .alignment_service import AlignmentService
    
    novel_path = Path(args.novel_path).resolve()
    
    if not novel_path.exists():
        logger.error(f"Novel path does not exist: {novel_path}")
        sys.exit(1)
    
    # Initialize components
    config = Config()
    db_manager = DatabaseManager(db_path=args.db_path)
    service = AlignmentService(config=config, db_manager=db_manager)
    
    try:
        result = service.align_novel(
            novel_root=str(novel_path),
            rebuild_embeddings=args.rebuild,
            embedding_model=args.model,
            min_confidence=args.min_confidence,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("ALIGNMENT COMPLETE")
        print("=" * 60)
        print(f"Novel: {result['novel']}")
        print(f"Matched: {result['statistics']['matched']}")
        print(f"Unmatched JP: {result['statistics']['unmatched_jp']}")
        print(f"Unmatched EN: {result['statistics']['unmatched_en']}")
        print(f"Average Confidence: {result['statistics']['average_confidence']:.4f}")
        print("=" * 60)
        
        # Save graph if output path specified
        if args.output:
            from .graph_builder import GraphBuilder
            builder = GraphBuilder(result['novel'])
            builder.save_to_file(result['graph'], args.output)
            print(f"\nGraph saved to: {args.output}")
        
        return result
        
    except Exception as e:
        logger.error(f"Alignment failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_rebuild_embeddings(args):
    """Rebuild embeddings for a novel."""
    from .config import Config
    from .database.db_manager import DatabaseManager
    from .alignment_service import AlignmentService
    
    novel_path = Path(args.novel_path).resolve()
    
    if not novel_path.exists():
        logger.error(f"Novel path does not exist: {novel_path}")
        sys.exit(1)
    
    config = Config()
    db_manager = DatabaseManager(db_path=args.db_path)
    service = AlignmentService(config=config, db_manager=db_manager)
    
    try:
        result = service.align_novel(
            novel_root=str(novel_path),
            rebuild_embeddings=True,
            embedding_model=args.model,
        )
        
        print(f"\nEmbeddings rebuilt successfully!")
        print(f"Novel: {result['novel']}")
        print(f"Total matches: {result['statistics']['matched']}")
        
    except Exception as e:
        logger.error(f"Rebuild failed: {e}", exc_info=True)
        sys.exit(1)


def cmd_export_graph(args):
    """Export alignment graph to JSON file."""
    from .database.db_manager import DatabaseManager
    from .graph_builder import GraphBuilder
    
    db_manager = DatabaseManager(db_path=args.db_path)
    
    # Find novel by name or path
    novel_name = Path(args.novel_path).name
    novel = db_manager.get_novel_by_name(novel_name)
    
    if not novel:
        logger.error(f"Novel not found: {novel_name}")
        sys.exit(1)
    
    # Get relations
    relations = db_manager.get_relations_by_novel(novel['id'])
    
    # Build graph
    builder = GraphBuilder(novel['name'])
    
    # Get chapters for unmatched lists
    all_chapters = db_manager.get_chapters_by_novel(novel['id'])
    jp_chapters = [c for c in all_chapters if c['language'] == 'jp']
    en_chapters = [c for c in all_chapters if c['language'] == 'en']
    
    matched_jp = {r['jp_path'] for r in relations}
    matched_en = {r['en_path'] for r in relations}
    
    graph = {
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
    }
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    
    print(f"Graph exported to: {output_path}")
    print(f"Relations: {len(graph['relations'])}")
    print(f"Unmatched JP: {len(graph['unmatched_jp'])}")
    print(f"Unmatched EN: {len(graph['unmatched_en'])}")


def cmd_validate(args):
    """Validate alignment results."""
    from .database.db_manager import DatabaseManager
    from .graph_builder import GraphBuilder, format_confidence
    
    db_manager = DatabaseManager(db_path=args.db_path)
    
    novel_name = Path(args.novel_path).name
    novel = db_manager.get_novel_by_name(novel_name)
    
    if not novel:
        logger.error(f"Novel not found: {novel_name}")
        sys.exit(1)
    
    validation = db_manager.get_latest_validation(novel['id'])
    
    if not validation:
        print("No validation results found. Run alignment first.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Novel: {novel['name']}")
    print(f"Timestamp: {validation.get('run_timestamp', 'N/A')}")
    print("-" * 60)
    print(f"Matched Chapters: {validation['matched_count']}")
    print(f"Unmatched JP: {validation['unmatched_jp_count']}")
    print(f"Unmatched EN: {validation['unmatched_en_count']}")
    print(f"Average Confidence: {validation['average_confidence']:.4f}")
    print("-" * 60)
    
    details = validation.get('details_json', {})
    if details:
        dist = details.get('confidence_distribution', {})
        print("\nConfidence Distribution:")
        print(f"  Very Strong (0.95-1.00): {dist.get('very_strong', 0)}")
        print(f"  Strong (0.80-0.95):      {dist.get('strong', 0)}")
        print(f"  Possible (0.50-0.80):    {dist.get('possible', 0)}")
        print(f"  Weak (0.00-0.50):        {dist.get('weak', 0)}")
        
        stats = details.get('statistics', {})
        print("\nCoverage:")
        print(f"  JP Coverage: {stats.get('jp_chapter_coverage', 0):.1f}%")
        print(f"  EN Coverage: {stats.get('en_chapter_coverage', 0):.1f}%")
    
    print("=" * 60)


def cmd_stats(args):
    """Show statistics for a novel."""
    from .database.db_manager import DatabaseManager
    
    db_manager = DatabaseManager(db_path=args.db_path)
    
    novel_name = Path(args.novel_path).name
    novel = db_manager.get_novel_by_name(novel_name)
    
    if not novel:
        logger.error(f"Novel not found: {novel_name}")
        sys.exit(1)
    
    stats = db_manager.get_statistics(novel['id'])
    
    print("\n" + "=" * 60)
    print("NOVEL STATISTICS")
    print("=" * 60)
    print(f"Novel ID: {stats['novel_id']}")
    print(f"Japanese Chapters: {stats['jp_chapters']}")
    print(f"English Chapters: {stats['en_chapters']}")
    print(f"Total Relations: {stats['total_relations']}")
    print(f"Average Confidence: {stats['average_confidence']:.4f}")
    print("=" * 60)


def cmd_server(args):
    """Start the API server."""
    import uvicorn
    from .config import get_config
    from .database.db_manager import DatabaseManager
    from .alignment_service import AlignmentService
    from .api.fastapi_server import app, set_dependencies
    
    config = get_config()
    db_manager = DatabaseManager(db_path=config.database.db_path)
    service = AlignmentService(config=config, db_manager=db_manager)
    
    # Set dependencies
    set_dependencies(db_manager, service)
    
    print(f"\nStarting API server on {config.api.host}:{config.api.port}")
    print("API Documentation: http://localhost:8000/docs\n")
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Novel Aligner - Automatic JP/EN Chapter Alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Align command
    align_parser = subparsers.add_parser('align', help='Align chapters in a novel')
    align_parser.add_argument('novel_path', help='Path to novel directory')
    align_parser.add_argument('--db-path', default='novel_aligner.db', help='Database path')
    align_parser.add_argument('--output', '-o', help='Output JSON file path')
    align_parser.add_argument('--rebuild', action='store_true', help='Rebuild embeddings')
    align_parser.add_argument('--model', help='Embedding model to use')
    align_parser.add_argument('--min-confidence', type=float, default=0.3, help='Minimum confidence threshold')
    align_parser.set_defaults(func=cmd_align)
    
    # Rebuild embeddings command
    rebuild_parser = subparsers.add_parser('rebuild-embeddings', help='Rebuild embeddings')
    rebuild_parser.add_argument('novel_path', help='Path to novel directory')
    rebuild_parser.add_argument('--db-path', default='novel_aligner.db', help='Database path')
    rebuild_parser.add_argument('--model', help='Embedding model to use')
    rebuild_parser.set_defaults(func=cmd_rebuild_embeddings)
    
    # Export graph command
    export_parser = subparsers.add_parser('export-graph', help='Export alignment graph')
    export_parser.add_argument('novel_path', help='Path to novel directory or novel name')
    export_parser.add_argument('output', help='Output JSON file path')
    export_parser.add_argument('--db-path', default='novel_aligner.db', help='Database path')
    export_parser.set_defaults(func=cmd_export_graph)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate alignment results')
    validate_parser.add_argument('novel_path', help='Path to novel directory or novel name')
    validate_parser.add_argument('--db-path', default='novel_aligner.db', help='Database path')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show novel statistics')
    stats_parser.add_argument('novel_path', help='Path to novel directory or novel name')
    stats_parser.add_argument('--db-path', default='novel_aligner.db', help='Database path')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--debug', action='store_true', help='Debug mode')
    server_parser.set_defaults(func=cmd_server)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Update config from args if server command
    if args.command == 'server':
        from .config import DEFAULT_CONFIG
        DEFAULT_CONFIG.api.host = args.host
        DEFAULT_CONFIG.api.port = args.port
        DEFAULT_CONFIG.api.debug = args.debug
    
    args.func(args)


if __name__ == '__main__':
    main()
