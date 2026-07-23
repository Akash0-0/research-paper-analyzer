from __future__ import annotations
import logging
from utils.schemas import Analysis
from utils.llm_client import llm_call, llm_call_with_feedback
from utils.prompt_loader import load

logger = logging.getLogger(__name__)


def run(state: dict) -> dict:
    text, fb = state.get("paper_text", ""), state.get("review_feedback", {}).get("analyzer", "")
    prev = state.get("analysis")
    if not text:
        return {**state, "errors": [*state.get("errors", []), "No text for analyzer"]}
    prompt = load("analyzer")
    window = text[:15000]
    if prev and fb:
        result: Analysis = llm_call_with_feedback(prompt, f"Analyze this paper:\n\n{window}", Analysis, prev.model_dump_json(indent=2), fb)
    else:
        result: Analysis = llm_call(prompt, f"Analyze this paper:\n\n{window}", Analysis)
    return {**state, "analysis": result, "stage": "analysis_done", "analysis_approved": False}
