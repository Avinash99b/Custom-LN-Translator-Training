from typing import Dict, Any, List, Tuple

class GraphBuilder:
    @staticmethod
    def build_graph(novel_name: str, jp_files: List[str], en_files: List[str], alignments: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        relations = []
        matched_jp = set()
        matched_en = set()

        for jp_idx, en_idx, confidence in alignments:
            jp_file = jp_files[jp_idx]
            en_file = en_files[en_idx]
            matched_jp.add(jp_file)
            matched_en.add(en_file)
            
            relations.append({
                "jp": jp_file,
                "en": en_file,
                "confidence": round(confidence, 4),
                "strength": GraphBuilder._get_strength(confidence)
            })

        unmatched_jp = [f for f in jp_files if f not in matched_jp]
        unmatched_en = [f for f in en_files if f not in matched_en]

        return {
            "novel": novel_name,
            "relations": relations,
            "unmatched_jp": unmatched_jp,
            "unmatched_en": unmatched_en,
            "stats": {
                "matched_count": len(relations),
                "unmatched_jp_count": len(unmatched_jp),
                "unmatched_en_count": len(unmatched_en),
                "average_confidence": round(sum(r['confidence'] for r in relations) / len(relations), 4) if relations else 0.0
            }
        }

    @staticmethod
    def _get_strength(confidence: float) -> str:
        if confidence >= 0.95: return "very strong"
        if confidence >= 0.80: return "strong"
        if confidence >= 0.50: return "possible"
        return "weak"