"""
Template Loader - Loads unstructured user context for plan/contract generation
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

ALLOWED_EXTENSIONS = {'.txt', '.md', '.markdown', '.rst', '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.conf', '.log', '.csv', '.tsv'}
DEFAULT_MAX_FILE_SIZE_KB = 50
DEFAULT_MAX_TOTAL_SIZE_KB = 200

class TemplateLoader:
    def __init__(self, templates_dir: str, max_file_size_kb: int = DEFAULT_MAX_FILE_SIZE_KB, max_total_size_kb: int = DEFAULT_MAX_TOTAL_SIZE_KB):
        self.templates_dir = Path(templates_dir)
        self.max_file_size_kb = max_file_size_kb
        self.max_total_size_kb = max_total_size_kb
        self.warnings = []
        self.loaded_templates = {}

    def is_valid_text_file(self, filepath: Path) -> Tuple[bool, str]:
        if filepath.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False, f"Extension '{filepath.suffix}' not in whitelist"
        try:
            size_kb = filepath.stat().st_size / 1024
            if size_kb > self.max_file_size_kb:
                return False, f"File size {size_kb:.1f}KB exceeds {self.max_file_size_kb}KB limit"
            with open(filepath, 'r', encoding='utf-8') as f:
                f.read()
            return True, "Valid"
        except Exception as e:
            return False, str(e)

    def load_all(self) -> Dict[str, str]:
        if not self.templates_dir.exists() or not self.templates_dir.is_dir():
            return {}
        valid_files = sorted([f for f in self.templates_dir.iterdir() if not f.is_dir() and self.is_valid_text_file(f)[0]], key=lambda x: x.name)
        total_size_kb = 0
        for filepath in valid_files:
            try:
                content = filepath.read_text(encoding='utf-8')
                if not content.strip(): continue
                sz = len(content.encode('utf-8')) / 1024
                if total_size_kb + sz > self.max_total_size_kb: continue
                self.loaded_templates[filepath.name] = content
                total_size_kb += sz
            except: pass
        return self.loaded_templates

    def format_for_prompt(self) -> Optional[str]:
        if not self.loaded_templates: return None
        lines = ["<user_provided_context>", "The user has provided unstructured context files.", ""]
        for name, content in self.loaded_templates.items():
            lines.extend([f"--- FILE: {name} ---", content, f"--- END: {name} ---", ""])
        lines.append("</user_provided_context>")
        return "\n".join(lines)

def load_templates(templates_dir, verbose=False):
    loader = TemplateLoader(templates_dir)
    loader.load_all()
    return loader.format_for_prompt(), loader.warnings
