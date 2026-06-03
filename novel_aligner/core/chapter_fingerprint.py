"""
Chapter Fingerprint Module
"""
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChapterFingerprint:
    file_path: str
    title: Optional[str] = None
    summary: Optional[str] = None
    characters: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    embedding_mode: str = "raw"
    hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file': self.file_path, 'title': self.title, 'summary': self.summary,
            'characters': self.characters, 'events': self.events,
            'embedding': self.embedding, 'embedding_mode': self.embedding_mode, 'hash': self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChapterFingerprint":
        return cls(
            file_path=data.get('file', ''), title=data.get('title'), summary=data.get('summary'),
            characters=data.get('characters', []), events=data.get('events', []),
            embedding=data.get('embedding'), embedding_mode=data.get('embedding_mode', 'raw'),
            hash=data.get('hash', ''),
        )

class FingerprintGenerator:
    HONORIFICS_JP = ['さん', 'くん', 'ちゃん', '先生', '様', 'さま']
    TITLES_EN = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Sir', 'Lady', 'Lord']
    EVENT_PATTERNS = [
        r'(?:killed|defeated|fought|met|encountered|discovered|found|lost|gained|learned)',
        r'(?:戦った | 倒した | 出会った | 見つけた | 失った | 手に入れた | 学んだ)',
    ]
    
    def __init__(self, sample_chars: int = 2000, sample_end_chars: int = 1000,
                 max_title_length: int = 100, use_summary: bool = True, embedding_mode: str = "summary"):
        self.sample_chars = sample_chars
        self.sample_end_chars = sample_end_chars
        self.max_title_length = max_title_length
        self.use_summary = use_summary
        self.embedding_mode = embedding_mode
        
        # Build patterns using concatenation to avoid format() issues with regex braces
        honorific_pattern = '|'.join(self.HONORIFICS_JP)
        self.name_patterns_jp = [
            re.compile(r'([一-\u9fff]' + r'{2,4})' + r'(?:' + honorific_pattern + r')'),
            re.compile(r'([ァ - ヴー]' + r'{3,6})' + r'(?:' + honorific_pattern + r')'),
        ]
        
        titles_pattern = '|'.join(self.TITLES_EN)
        self.name_patterns_en = [
            re.compile(r'\b([A-Z][a-z]+)\s+(?:' + titles_pattern + r')\b'),
            re.compile(r'\b(?:' + titles_pattern + r')\s+([A-Z][a-z]+)\b'),
        ]
        
        self.nlp_en = None
        self.nlp_ja = None
        try:
            import spacy
            try: self.nlp_en = spacy.load('en_core_web_sm')
            except OSError: pass
            try: self.nlp_ja = spacy.load('ja_core_news_sm')
            except OSError: pass
        except ImportError: pass
    
    def generate(self, content: str, title: Optional[str] = None, language: str = 'en', file_path: str = '') -> ChapterFingerprint:
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        normalized_title = self._normalize_title(title) if title else None
        sample_text = self._extract_sample(content)
        characters = self._extract_characters(sample_text, language)
        events = self._extract_events(sample_text, language)
        summary = self._generate_summary(sample_text, language) if self.use_summary else None
        
        return ChapterFingerprint(
            file_path=file_path, title=normalized_title, summary=summary,
            characters=characters, events=events, embedding=None,
            embedding_mode=self.embedding_mode, hash=content_hash,
        )
    
    def _normalize_title(self, title: str) -> str:
        if not title: return ""
        if len(title) > self.max_title_length: title = title[:self.max_title_length]
        for pattern in [r'^Chapter\s+\d+[:\s]*', r'^第\s*\d+\s*[話回章][：:\s]*', r'^Vol\.\s*\d+[:\s]*']:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        return ' '.join(title.split()).strip(' -:;,.!?')
    
    def _extract_sample(self, content: str) -> str:
        lines = content.split('\n')
        start_idx = next((i for i, line in enumerate(lines) if len(line.strip()) > 10), 0)
        sample_parts, char_count = [], 0
        for line in lines[start_idx:]:
            if char_count >= self.sample_chars: break
            sample_parts.append(line)
            char_count += len(line)
        end_parts, end_count = [], 0
        for line in reversed(lines[-50:]):
            if end_count >= self.sample_end_chars: break
            end_parts.insert(0, line)
            end_count += len(line)
        sample = '\n'.join(sample_parts)
        if end_parts: sample += '\n...\n' + '\n'.join(end_parts)
        return sample
    
    def _extract_characters(self, text: str, language: str) -> List[str]:
        characters = set()
        if language == 'en' and self.nlp_en:
            for ent in self.nlp_en(text[:5000]).ents:
                if ent.label_ == 'PERSON': characters.add(ent.text)
        elif language == 'jp' and self.nlp_ja:
            for ent in self.nlp_ja(text[:5000]).ents:
                if ent.label_ in ('PERSON', 'PER'): characters.add(ent.text)
        
        for pattern in (self.name_patterns_jp if language == 'jp' else self.name_patterns_en):
            for match in pattern.findall(text):
                if isinstance(match, tuple): match = match[0]
                characters.add(match)
        
        return sorted([c for c in characters if 2 <= len(c) <= 20 and not c.isdigit()])[:20]
    
    def _extract_events(self, text: str, language: str) -> List[str]:
        events = []
        for sentence in re.split(r'[。.!?\\n]', text):
            for pattern in self.EVENT_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    if sentence.strip() and len(sentence) < 200:
                        events.append(sentence.strip())
                        break
        return events[:10]
    
    def _generate_summary(self, text: str, language: str) -> str:
        splitter = r'[。!?]' if language == 'jp' else r'[.!?]'
        sentences = [s.strip() for s in re.split(splitter, text) if s.strip()]
        summary_sentences = []
        for sentence in sentences[:5]:
            if len(sentence) > 20:
                summary_sentences.append(sentence)
            if len(' '.join(summary_sentences)) >= 300: break
        summary = '. '.join(summary_sentences)
        return (summary.replace('.', '') + '。') if language == 'jp' else (summary.rstrip('.') + '.')
