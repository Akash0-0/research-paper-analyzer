from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()
