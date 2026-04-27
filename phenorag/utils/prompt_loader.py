# phenorag/utils/prompt_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class PromptLoader:
    """
    Loads prompt templates from disk.

    For now: simple .txt with {placeholders}.
    Later: you can swap to Jinja2 with minimal changes.
    """

    def __init__(self, prompts_root: str | Path):
        self.prompts_root = Path(prompts_root)

    def load(self, prompt_id: str) -> str:
        """
        prompt_id examples:
          - "agent2/ra_relatedness_v1"
        resolved to:
          prompts_root / "agent2" / "ra_relatedness_v1.txt"
        """
        path = self.prompts_root / f"{prompt_id}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, prompt_id: str, **kwargs: Dict[str, Any]) -> str:
        template = self.load(prompt_id)
        return template.format(**kwargs)