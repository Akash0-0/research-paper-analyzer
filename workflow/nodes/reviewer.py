"""Reviewer — quality gate with score+feedback, max 2 retries per agent."""

from __future__ import annotations
import logging
from utils.schemas import ReviewScore
from utils.llm_client import llm_call
from utils.prompt_loader import load

logger = logging.getLogger(__name__)
MAX_RETRIES = 2


def _context(agent: str, text: str, output: str) -> str:
    sample = text[-10000:] if agent == "citation" else text[:8000]
    return f"Agent: {agent}\n\nPaper:\n{sample}\n\nOutput:\n{output}\n\nScore 1-10. <7 = reject with issues."


def _review(state: dict, agent: str) -> dict:
    key = {"analyzer": "analysis", "summary": "summary", "citation": "citations", "insights": "insights"}[agent]
    output_obj = state.get(key)
    if not output_obj:
        logger.warning("No %s output to review", agent)
        return state
    out_json = output_obj.model_dump_json(indent=2)
    review: ReviewScore = llm_call(load("reviewer"), _context(agent, state.get("paper_text", ""), out_json), ReviewScore)

    retries = dict(state.get("retry_counts", {}))
    feedback = dict(state.get("review_feedback", {}))
    scores = dict(state.get("review_scores", {}))
    r = retries.get(agent, 0)
    scores[agent] = review.score

    if review.score >= 7:
        feedback.pop(agent, None)
        upd = {"analysis_approved": True} if agent == "analyzer" else {}
        logger.info("%s APPROVED (%d/10)", agent, review.score)
        return {**state, "review_scores": scores, "review_feedback": feedback, "stage": f"{agent}_approved", **upd}

    if r >= MAX_RETRIES:
        scores[agent] = max(review.score, 5)
        feedback[agent] = f"Max retries. Last: {review.feedback}"
        upd = {"analysis_approved": True} if agent == "analyzer" else {}
        logger.warning("%s force-approved after %d retries", agent, MAX_RETRIES)
        return {**state, "review_scores": scores, "review_feedback": feedback, **upd, "stage": f"{agent}_forced"}

    retries[agent] = r + 1
    feedback[agent] = review.feedback + ("; " + "; ".join(review.issues) if review.issues else "")
    logger.info("%s REJECTED (%d/10) retry %d/%d", agent, review.score, r + 1, MAX_RETRIES)
    return {**state, "retry_counts": retries, "review_feedback": feedback, "review_scores": scores, "stage": f"{agent}_rejected"}


def analyzer(state: dict) -> dict: return _review(state, "analyzer")
def summary(state: dict) -> dict: return _review(state, "summary")
def citation(state: dict) -> dict: return _review(state, "citation")
def insights(state: dict) -> dict: return _review(state, "insights")
