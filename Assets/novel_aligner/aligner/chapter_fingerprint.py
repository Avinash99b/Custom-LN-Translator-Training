import re
from typing import Dict, Any, List
try:
    import spacy
    nlp = spacy.load("xx_ent_wiki_sm") # Multilingual NER
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class ChapterFingerprint:
    def __init__(self, embedding_engine, max_chars: int = 2000, mode: str = "summary"):
        self.engine = embedding_engine
        self.max_chars = max_chars
        self.mode = mode

    def extract(self, file_path: str, absolute_path: str) -> Dict[str, Any]:
        with open(absolute_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()

        lines = content.split('\n')
        title = lines[0] if lines else ""
        text_body = content[:self.max_chars]

        if self.mode == "summary":
            # In production, swap this for an LLM API call.
            summary = self._generate_heuristic_summary(text_body)
            target_text = f"{title}\n{summary}"
        else:
            summary = text_body[:500]
            target_text = text_body

        characters, events = self._extract_entities(target_text)
        embedding = self.engine.encode([target_text])[0]

        return {
            "file": file_path,
            "title": title,
            "summary": summary,
            "characters": characters,
            "events": events,
            "embedding": embedding
        }

    def _generate_heuristic_summary(self, text: str) -> str:
        # Simple extraction of first/last paragraphs to form a naive summary
        paragraphs = [p for p in text.split('\n\n') if len(p) > 20]
        if len(paragraphs) <= 2: return text
        return paragraphs[0] + "\n...\n" + paragraphs[-1]

    def _extract_entities(self, text: str) -> tuple[List[str], List[str]]:
        characters, events = set(), set()
        if SPACY_AVAILABLE:
            doc = nlp(text[:1000])
            for ent in doc.ents:
                if ent.label_ == "PER": characters.add(ent.text)
                elif ent.label_ in ["EVENT", "LOC"]: events.add(ent.text)
        else:
            # Fallback: Extract capitalized words > 3 chars (naive English/Romaji approach)
            words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
            characters.update(words[:10])
            
        return list(characters), list(events)