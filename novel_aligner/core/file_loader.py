"""
File Loader Module

Handles loading and parsing of chapter files from the novel directory structure.
Supports both Japanese and English chapters with various file formats.
"""
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChapterData:
    """
    Represents a single chapter with its content and metadata.
    
    Attributes:
        file_path: Absolute path to the chapter file
        relative_path: Path relative to novel root (e.g., "JP/1.txt")
        language: Language code ('jp' or 'en')
        chapter_number: Extracted chapter number if available
        title: Chapter title extracted from content
        content: Full chapter content
        content_preview: First N characters of content
    """
    file_path: str
    relative_path: str
    language: str
    chapter_number: Optional[int] = None
    title: Optional[str] = None
    content: str = ""
    content_preview: str = ""
    
    def __post_init__(self):
        """Generate preview after initialization."""
        if self.content and not self.content_preview:
            self.content_preview = self.content[:500]


class FileLoader:
    """
    Loads chapter files from a novel directory structure.
    
    Expected structure:
        NovelName/
        ├── JP/
        │   ├── 1.txt
        │   ├── 2.txt
        │   └── ...
        └── EN/
            ├── 1.txt
            ├── 2.txt
            └── ...
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.htm', '.html'}
    
    # Patterns for extracting chapter numbers from filenames
    CHAPTER_NUMBER_PATTERNS = [
        r'^(\d+)',                      # Simple number: "1.txt"
        r'[Cc]hapter[_\s]*(\d+)',       # "Chapter_1.txt" or "Chapter 1.txt"
        r'[第](\d+[話回章])',              # Japanese: "第1話.txt"
        r'(\d+)[巻章話回]',               # "1巻.txt", "1章.txt"
        r'side[_\s]*story[_\s]*(\d*)',  # Side stories
        r'extra[_\s]*(\d*)',            # Extra chapters
    ]
    
    # Patterns for extracting titles from file content
    TITLE_PATTERNS_JP = [
        r'^[#\s]*第\s*\d+\s*[話回章][：:\s]*(.+?)$',
        r'^[#\s]*(.+?)[のへは]\s*第\s*\d+',
        r'^[#\s]*(プロローグ|エピローグ|短編|外伝)[:：\s]*(.*)$',
    ]
    
    TITLE_PATTERNS_EN = [
        r'^[#\s]*Chapter\s+\d+[:\s]*(.+?)$',
        r'^[#\s]*(.+?)\s+-\s+Chapter\s+\d+',
        r'^[#\s]*(Prologue|Epilogue|Side Story|Extra)[:\s]*(.*)$',
    ]
    
    def __init__(self, novel_root: str, max_workers: int = 4):
        """
        Initialize the file loader.
        
        Args:
            novel_root: Path to the novel root directory
            max_workers: Maximum number of parallel file loaders
        """
        self.novel_root = Path(novel_root).resolve()
        self.max_workers = max_workers
        self.jp_chapters: List[ChapterData] = []
        self.en_chapters: List[ChapterData] = []
        
        if not self.novel_root.exists():
            raise FileNotFoundError(f"Novel root not found: {self.novel_root}")
    
    def load_all(self) -> Tuple[List[ChapterData], List[ChapterData]]:
        """
        Load all chapters from both JP and EN directories.
        
        Returns:
            Tuple of (jp_chapters, en_chapters)
        """
        logger.info(f"Loading chapters from {self.novel_root}")
        
        jp_dir = self.novel_root / "JP"
        en_dir = self.novel_root / "EN"
        
        # Load chapters in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            if jp_dir.exists():
                futures[executor.submit(self._load_directory, jp_dir, 'jp')] = 'jp'
            
            if en_dir.exists():
                futures[executor.submit(self._load_directory, en_dir, 'en')] = 'en'
            
            for future in as_completed(futures):
                lang = futures[future]
                try:
                    chapters = future.result()
                    if lang == 'jp':
                        self.jp_chapters = chapters
                    else:
                        self.en_chapters = chapters
                except Exception as e:
                    logger.error(f"Error loading {lang} chapters: {e}")
                    raise
        
        logger.info(f"Loaded {len(self.jp_chapters)} JP chapters and {len(self.en_chapters)} EN chapters")
        return self.jp_chapters, self.en_chapters
    
    def _load_directory(self, directory: Path, language: str) -> List[ChapterData]:
        """
        Load all chapter files from a directory.
        
        Args:
            directory: Directory path
            language: Language code ('jp' or 'en')
            
        Returns:
            List of ChapterData objects
        """
        chapters = []
        
        # Find all supported files
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"**/*{ext}"))  # Recursive
        
        # Remove duplicates and sort
        files = sorted(set(files), key=lambda x: self._extract_sort_key(x))
        
        for file_path in files:
            try:
                chapter = self._load_file(file_path, language)
                if chapter:
                    chapters.append(chapter)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Sort by chapter number if available, otherwise by filename
        chapters.sort(key=lambda c: (
            c.chapter_number if c.chapter_number is not None else float('inf'),
            c.file_path
        ))
        
        return chapters
    
    def _load_file(self, file_path: Path, language: str) -> Optional[ChapterData]:
        """
        Load a single chapter file.
        
        Args:
            file_path: Path to the file
            language: Language code
            
        Returns:
            ChapterData object or None if loading fails
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate relative path
            relative_path = str(file_path.relative_to(self.novel_root))
            
            # Extract chapter number from filename
            chapter_number = self._extract_chapter_number(file_path.name)
            
            # Extract title from content
            title = self._extract_title(content, language)
            
            # Generate preview
            preview_length = min(500, len(content))
            content_preview = content[:preview_length].strip()
            
            return ChapterData(
                file_path=str(file_path),
                relative_path=relative_path,
                language=language,
                chapter_number=chapter_number,
                title=title,
                content=content,
                content_preview=content_preview
            )
            
        except UnicodeDecodeError:
            # Try alternative encodings
            for encoding in ['shift_jis', 'cp932', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.warning(f"Could not decode {file_path} with any encoding")
                return None
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _extract_chapter_number(self, filename: str) -> Optional[int]:
        """
        Extract chapter number from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Chapter number or None
        """
        for pattern in self.CHAPTER_NUMBER_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    num_str = match.group(1)
                    if num_str:
                        return int(num_str)
                except (ValueError, IndexError):
                    continue
        
        # Try to extract just digits from filename
        digits = re.sub(r'\D', '', filename.split('.')[0])
        if digits:
            return int(digits)
        
        return None
    
    def _extract_title(self, content: str, language: str) -> Optional[str]:
        """
        Extract chapter title from content.
        
        Args:
            content: Chapter content
            language: Language code
            
        Returns:
            Title or None
        """
        patterns = self.TITLE_PATTERNS_JP if language == 'jp' else self.TITLE_PATTERNS_EN
        
        lines = content.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Get the last non-empty group
                    groups = [g for g in match.groups() if g and g.strip()]
                    if groups:
                        return groups[-1].strip()
        
        # If no title found, use first non-empty line as fallback
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:
                return line
        
        return None
    
    def _extract_sort_key(self, file_path: Path) -> Tuple:
        """
        Extract a sort key from file path for proper ordering.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple for sorting
        """
        filename = file_path.stem
        chapter_num = self._extract_chapter_number(filename)
        
        if chapter_num is not None:
            return (0, chapter_num, filename)
        else:
            return (1, 0, filename)
    
    def get_novel_name(self) -> str:
        """Get the name of the novel from the root directory."""
        return self.novel_root.name
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded chapters.
        
        Returns:
            Dictionary with statistics
        """
        jp_with_titles = sum(1 for c in self.jp_chapters if c.title)
        en_with_titles = sum(1 for c in self.en_chapters if c.title)
        jp_with_numbers = sum(1 for c in self.jp_chapters if c.chapter_number)
        en_with_numbers = sum(1 for c in self.en_chapters if c.chapter_number)
        
        return {
            'novel_name': self.get_novel_name(),
            'jp_chapters': len(self.jp_chapters),
            'en_chapters': len(self.en_chapters),
            'jp_with_titles': jp_with_titles,
            'en_with_titles': en_with_titles,
            'jp_with_numbers': jp_with_numbers,
            'en_with_numbers': en_with_numbers,
            'total_content_size_jp': sum(len(c.content) for c in self.jp_chapters),
            'total_content_size_en': sum(len(c.content) for c in self.en_chapters),
        }
