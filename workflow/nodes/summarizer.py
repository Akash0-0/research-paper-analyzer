from __future__ import annotations
import logging
from utils.schemas import Summary
from utils.llm_client import llm_call, llm_call_with_feedback
from utils.prompt_loader import load

logger = logging.getLogger(__name__)


def run(state: dict) -> dict:
    text, fb = state.get("paper_text", ""), state.get("review_feedback", {}).get("summary", "")
    prev, analysis = state.get("summary"), state.get("analysis")
    if not text:
        return {**state, "errors": [*state.get("errors", []), "No text for summarizer"]}
    prompt = load("summarizer")
    ctx = f"\nAnalysis context:\nProblem: {analysis.problem}\nFindings: {analysis.findings}" if analysis else ""
    window = text[:12000]
    if prev and fb:
        result: Summary = llm_call_with_feedback(prompt, f"Summarize:\n\n{window}{ctx}", Summary, prev.model_dump_json(indent=2), fb)
    else:
        result: Summary = llm_call(prompt, f"Summarize:\n\n{window}{ctx}", Summary)
    return {**state, "summary": result, "stage": "summary_done"}
