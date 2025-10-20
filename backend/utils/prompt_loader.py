from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from ..config.settings import settings  # optional

def _candidates(filename: str):
    if settings is not None:
        d = getattr(settings, "prompts_dir", None) or getattr(settings, "PROMPTS_DIR", None)
        if d:
            yield Path(d).expanduser().resolve() / filename
    # repo root / prompts
    yield Path(__file__).resolve().parents[1] / "prompts" / filename
    # CWD / prompts
    yield Path.cwd() / "prompts" / filename

@lru_cache(maxsize=64)
def load_prompt(filename: str, *, encoding="utf-8") -> str:
    tried = []
    for p in _candidates(filename):
        tried.append(str(p))
        if p.is_file():
            return p.read_text(encoding=encoding)
    raise FileNotFoundError(f"Prompt '{filename}' not found. Looked in:\n" + "\n".join(tried))
