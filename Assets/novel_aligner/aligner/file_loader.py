import os
import re
from pathlib import Path
from typing import List

class FileLoader:
    @staticmethod
    def _natural_sort_key(s: str) -> List[Any]:
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

    def load_chapters(self, base_path: str, language: str) -> List[str]:
        target_dir = Path(base_path) / language
        if not target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {target_dir}")
        
        files = [str(p.relative_to(base_path)) for p in target_dir.rglob("*.txt")]
        return sorted(files, key=self._natural_sort_key)