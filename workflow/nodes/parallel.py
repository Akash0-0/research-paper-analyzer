"""
Parallel sub-agents node — runs summary, citation, and insights
concurrently with their own review + retry loops.
"""

from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from workflow.nodes.summarizer import run as run_summarizer
from workflow.nodes.citation import run as run_citation
from workflow.nodes.insights import run as run_insights
from workflow.nodes.reviewer import summary as review_summary
from workflow.nodes.reviewer import citation as review_citation
from workflow.nodes.reviewer import insights as review_insights

logger = logging.getLogger(__name__)
MAX_R = 2


def _run_one(agent_fn, review_fn, agent_name: str, state: dict) -> dict:
    """Run agent + review in a loop. Mutates `state` dict directly."""
    for attempt in range(MAX_R + 1):
        try:
            result = agent_fn(state)
            state.update(result)  # merge agent output
            logger.info("%s attempt %d succeeded", agent_name, attempt + 1)
        except Exception as e:
            logger.error("%s failed (attempt %d/%d): %s", agent_name, attempt + 1, MAX_R, e)
            if attempt < MAX_R:
                continue
            break

        # Review
        try:
            review_result = review_fn(state)
            state.update(review_result)
            score = state.get("review_scores", {}).get(agent_name, 0)
            logger.info("%s review score: %d/10", agent_name, score)
            if score >= 7:
                logger.info("%s APPROVED (%d/10)", agent_name, score)
                return state
            logger.info("%s REJECTED (%d/10) retry %d/%d", agent_name, score, attempt + 1, MAX_R)
        except Exception as e:
            logger.error("%s review failed (attempt %d/%d): %s", agent_name, attempt + 1, MAX_R, e)

    logger.warning("%s exhausted retries", agent_name)
    return state


def run(state: dict) -> dict:
    """Execute summary, citation, insights in parallel threads."""
    shared = dict(state)  # copy for thread safety

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_run_one, run_summarizer, review_summary, "summary", shared): "summary",
            pool.submit(_run_one, run_citation, review_citation, "citation", shared): "citation",
            pool.submit(_run_one, run_insights, review_insights, "insights", shared): "insights",
        }

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                logger.info("✅ %s completed", name)
            except Exception as e:
                logger.error("💥 %s crashed: %s", name, e)

    return shared
