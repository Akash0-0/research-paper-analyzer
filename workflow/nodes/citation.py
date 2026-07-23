from __future__ import annotations
import logging
from utils.schemas import CitationOutput
from utils.llm_client import llm_call, llm_call_with_feedback
from utils.prompt_loader import load

logger = logging.getLogger(__name__)


def run(state: dict) -> dict:
    text, fb = state.get("paper_text", ""), state.get("review_feedback", {}).get("citation", "")
    prev = state.get("citations")
    if not text:
        return {**state, "errors": [*state.get("errors", []), "No text for citation extractor"]}
    prompt = load("citation")
    if prev and fb:
        result: CitationOutput = llm_call_with_feedback(prompt, f"Extract citations from:\n\n{text}", CitationOutput, prev.model_dump_json(indent=2), fb)
    else:
        result: CitationOutput = llm_call(prompt, f"Extract citations from:\n\n{text}", CitationOutput)
    return {**state, "citations": result, "stage": "citations_done"}
