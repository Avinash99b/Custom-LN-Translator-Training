"""
Graph Builder Module

Builds the final JSON graph output from alignment results.
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .core.sequence_aligner import AlignmentResult, AlignmentMatch
from .core.chapter_fingerprint import ChapterFingerprint
from .core.file_loader import ChapterData

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds the final JSON graph representation of chapter alignments.
    
    Output format:
    {
        "novel": "NovelName",
        "relations": [
            {
                "jp": "JP/1.txt",
                "en": "EN/2.txt",
                "confidence": 0.98
            }
        ],
        "unmatched_jp": ["JP/55.txt"],
        "unmatched_en": ["EN/1.txt", "EN/56.txt"]
    }
    """
    
    def __init__(self, novel_name: str):
        """
        Initialize the graph builder.
        
        Args:
            novel_name: Name of the novel
        """
        self.novel_name = novel_name
    
    def build(
        self,
        alignment_result: AlignmentResult,
        jp_chapters: List[ChapterData],
        en_chapters: List[ChapterData],
        jp_fingerprints: Optional[List[ChapterFingerprint]] = None,
        en_fingerprints: Optional[List[ChapterFingerprint]] = None,
    ) -> Dict[str, Any]:
        """
        Build the complete graph from alignment results.
        
        Args:
            alignment_result: Result from sequence alignment
            jp_chapters: List of Japanese chapters
            en_chapters: List of English chapters
            jp_fingerprints: Optional fingerprints for additional metadata
            en_fingerprints: Optional fingerprints for additional metadata
            
        Returns:
            Dictionary representing the complete graph
        """
        # Build relations list
        relations = []
        for match in alignment_result.matches:
            relation = {
                'jp': jp_chapters[match.jp_idx].relative_path,
                'en': en_chapters[match.en_idx].relative_path,
                'confidence': round(match.confidence, 4),
            }
            
            # Add titles if available
            jp_title = jp_chapters[match.jp_idx].title
            en_title = en_chapters[match.en_idx].title
            if jp_title or en_title:
                relation['jp_title'] = jp_title
                relation['en_title'] = en_title
            
            # Add fingerprint data if available
            if jp_fingerprints and match.jp_idx < len(jp_fingerprints):
                fp = jp_fingerprints[match.jp_idx]
                if fp.characters:
                    relation['jp_characters'] = fp.characters[:5]  # Top 5
                if fp.events:
                    relation['jp_events'] = fp.events[:3]  # Top 3
            
            if en_fingerprints and match.en_idx < len(en_fingerprints):
                fp = en_fingerprints[match.en_idx]
                if fp.characters:
                    relation['en_characters'] = fp.characters[:5]
                if fp.events:
                    relation['en_events'] = fp.events[:3]
            
            relations.append(relation)
        
        # Sort by confidence descending
        relations.sort(key=lambda r: r['confidence'], reverse=True)
        
        # Build unmatched lists
        unmatched_jp = [
            jp_chapters[idx].relative_path 
            for idx in alignment_result.unmatched_jp
        ]
        
        unmatched_en = [
            en_chapters[idx].relative_path 
            for idx in alignment_result.unmatched_en
        ]
        
        # Build metadata
        metadata = {
            'total_jp_chapters': len(jp_chapters),
            'total_en_chapters': len(en_chapters),
            'matched_count': len(relations),
            'unmatched_jp_count': len(unmatched_jp),
            'unmatched_en_count': len(unmatched_en),
            'average_confidence': round(alignment_result.average_confidence, 4),
        }
        
        return {
            'novel': self.novel_name,
            'relations': relations,
            'unmatched_jp': unmatched_jp,
            'unmatched_en': unmatched_en,
            'metadata': metadata,
        }
    
    def build_minimal(
        self,
        alignment_result: AlignmentResult,
        jp_chapters: List[ChapterData],
        en_chapters: List[ChapterData],
    ) -> Dict[str, Any]:
        """
        Build a minimal graph without additional metadata.
        
        Useful for quick exports and API responses.
        """
        relations = [
            {
                'jp': jp_chapters[m.jp_idx].relative_path,
                'en': en_chapters[m.en_idx].relative_path,
                'confidence': round(m.confidence, 4),
            }
            for m in alignment_result.matches
        ]
        
        relations.sort(key=lambda r: r['confidence'], reverse=True)
        
        return {
            'novel': self.novel_name,
            'relations': relations,
            'unmatched_jp': [
                jp_chapters[idx].relative_path 
                for idx in alignment_result.unmatched_jp
            ],
            'unmatched_en': [
                en_chapters[idx].relative_path 
                for idx in alignment_result.unmatched_en
            ],
        }
    
    def to_json(
        self,
        graph: Dict[str, Any],
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> str:
        """Convert graph to JSON string."""
        return json.dumps(graph, indent=indent, ensure_ascii=ensure_ascii)
    
    def save_to_file(
        self,
        graph: Dict[str, Any],
        output_path: str,
        indent: int = 2,
    ) -> None:
        """Save graph to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Graph saved to {output_path}")
    
    def generate_validation_report(
        self,
        alignment_result: AlignmentResult,
        jp_chapters: List[ChapterData],
        en_chapters: List[ChapterData],
    ) -> Dict[str, Any]:
        """
        Generate a validation report.
        
        Example output:
        {
            "matched": 285,
            "unmatched": 12,
            "average_confidence": 0.94,
            "confidence_distribution": {
                "very_strong": 200,
                "strong": 70,
                "possible": 15,
                "weak": 0
            },
            "details": {...}
        }
        """
        # Confidence distribution
        very_strong = 0  # 0.95-1.00
        strong = 0       # 0.80-0.95
        possible = 0     # 0.50-0.80
        weak = 0         # 0.00-0.50
        
        for match in alignment_result.matches:
            conf = match.confidence
            if conf >= 0.95:
                very_strong += 1
            elif conf >= 0.80:
                strong += 1
            elif conf >= 0.50:
                possible += 1
            else:
                weak += 1
        
        return {
            'matched': len(alignment_result.matches),
            'unmatched_jp': len(alignment_result.unmatched_jp),
            'unmatched_en': len(alignment_result.unmatched_en),
            'total_unmatched': len(alignment_result.unmatched_jp) + len(alignment_result.unmatched_en),
            'average_confidence': round(alignment_result.average_confidence, 4),
            'confidence_distribution': {
                'very_strong': very_strong,
                'strong': strong,
                'possible': possible,
                'weak': weak,
            },
            'statistics': {
                'jp_chapter_coverage': round(
                    len(alignment_result.matches) / len(jp_chapters) * 100, 2
                ) if jp_chapters else 0,
                'en_chapter_coverage': round(
                    len(alignment_result.matches) / len(en_chapters) * 100, 2
                ) if en_chapters else 0,
            },
        }


def format_confidence(confidence: float) -> str:
    """Format confidence as human-readable string."""
    if confidence >= 0.95:
        return "very strong"
    elif confidence >= 0.80:
        return "strong"
    elif confidence >= 0.50:
        return "possible"
    else:
        return "weak"
