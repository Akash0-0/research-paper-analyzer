"""
LangGraph workflow — the full research analyzer pipeline.

START → plan → analyzer → review_analyzer ──┬── (rejected, retries left) → analyzer
                                              └── (approved / max retries)
                                                    │
                                                    ▼
                                               summary ── retry ↻
                                                    │
                                                    ▼
                                              review_summary ──┬── (rejected, retries left) → summary
                                                                 └── (approved / max retries)
                                                                       │
                                                                       ▼
                                                                  citation ── retry ↻
                                                                       │
                                                                       ▼
                                                                 review_citation ──┬── retry → citation
                                                                                    └── done
                                                                                          │
                                                                                          ▼
                                                                                     insights ── retry ↻
                                                                                          │
                                                                                          ▼
                                                                                    review_insights ──┬── retry → insights
                                                                                                       └── done → combine → END
"""

from __future__ import annotations
import logging
from typing import Literal
from langgraph.graph import END, StateGraph
from workflow.state import WorkflowState
from workflow.nodes.boss import plan as boss_plan, combine as boss_combine
from workflow.nodes.analyzer import run as run_analyzer
from workflow.nodes.summarizer import run as run_summarizer
from workflow.nodes.citation import run as run_citation
from workflow.nodes.insights import run as run_insights
from workflow.nodes.reviewer import (
    analyzer as review_analyzer,
    summary as review_summary,
    citation as review_citation,
    insights as review_insights,
)

logger = logging.getLogger(__name__)
MAX_R = 2


def _route_analyzer(state: dict) -> Literal["analyzer", "fan_out"]:
    if state.get("analysis_approved") or state.get("retry_counts", {}).get("analyzer", 0) >= MAX_R:
        return "fan_out"
    return "analyzer"


def _route_summary(state: dict) -> Literal["summary", "done"]:
    if state.get("review_scores", {}).get("summary", 0) >= 7 or state.get("retry_counts", {}).get("summary", 0) >= MAX_R:
        return "done"
    return "summary"


def _route_citation(state: dict) -> Literal["citation", "done"]:
    if state.get("review_scores", {}).get("citation", 0) >= 7 or state.get("retry_counts", {}).get("citation", 0) >= MAX_R:
        return "done"
    return "citation"


def _route_insights(state: dict) -> Literal["insights", "done"]:
    if state.get("review_scores", {}).get("insights", 0) >= 7 or state.get("retry_counts", {}).get("insights", 0) >= MAX_R:
        return "done"
    return "insights"


def build() -> StateGraph:
    wf = StateGraph(WorkflowState)
    wf.add_node("plan", boss_plan)
    wf.add_node("analyzer", run_analyzer)
    wf.add_node("review_analyzer", review_analyzer)
    wf.add_node("summary", run_summarizer)
    wf.add_node("review_summary", review_summary)
    wf.add_node("citation", run_citation)
    wf.add_node("review_citation", review_citation)
    wf.add_node("insights", run_insights)
    wf.add_node("review_insights", review_insights)
    wf.add_node("combine", boss_combine)

    wf.set_entry_point("plan")
    wf.add_edge("plan", "analyzer")
    wf.add_edge("analyzer", "review_analyzer")
    wf.add_conditional_edges(
        "review_analyzer", _route_analyzer,
        {"analyzer": "analyzer", "fan_out": "summary"},
    )

    wf.add_edge("summary", "review_summary")
    wf.add_conditional_edges(
        "review_summary", _route_summary,
        {"summary": "summary", "done": "citation"},
    )

    wf.add_edge("citation", "review_citation")
    wf.add_conditional_edges(
        "review_citation", _route_citation,
        {"citation": "citation", "done": "insights"},
    )

    wf.add_edge("insights", "review_insights")
    wf.add_conditional_edges(
        "review_insights", _route_insights,
        {"insights": "insights", "done": "combine"},
    )

    wf.add_edge("combine", END)

    return wf.compile()
