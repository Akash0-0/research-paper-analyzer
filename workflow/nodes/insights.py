from __future__ import annotations
import logging
from utils.schemas import InsightOutput
from utils.llm_client import llm_call, llm_call_with_feedback
from utils.prompt_loader import load

logger = logging.getLogger(__name__)


def run(state: dict) -> dict:
    text, fb = state.get("paper_text", ""), state.get("review_feedback", {}).get("insights", "")
    prev, analysis = state.get("insights"), state.get("analysis")
    if not text:
        return {**state, "errors": [*state.get("errors", []), "No text for insights agent"]}
    prompt = load("insights")
    ctx = f"\nFindings: {analysis.findings}" if analysis else ""
    window = text[:12000]
    if prev and fb:
        result: InsightOutput = llm_call_with_feedback(prompt, f"Extract insights from:\n\n{window}{ctx}", InsightOutput, prev.model_dump_json(indent=2), fb)
    else:
        result: InsightOutput = llm_call(prompt, f"Extract insights from:\n\n{window}{ctx}", InsightOutput)
    return {**state, "insights": result, "stage": "insights_done"}
