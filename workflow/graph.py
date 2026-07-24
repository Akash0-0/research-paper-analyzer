"""
LangGraph workflow — the full research analyzer pipeline.

START → plan → analyzer ── retry (via review_analyzer)
                    └── approved → parallel_agents ──► combine → END
                                  (summary,citation,insights run concurrently)
"""

from __future__ import annotations
import logging
from typing import Literal
from langgraph.graph import END, StateGraph
from workflow.state import WorkflowState
from workflow.nodes.boss import plan as boss_plan, combine as boss_combine
from workflow.nodes.analyzer import run as run_analyzer
from workflow.nodes.parallel import run as run_parallel
from workflow.nodes.reviewer import analyzer as review_analyzer

logger = logging.getLogger(__name__)
MAX_R = 2


def _route_analyzer(state: dict) -> Literal["analyzer", "fan_out"]:
    if state.get("analysis_approved") or state.get("retry_counts", {}).get("analyzer", 0) >= MAX_R:
        return "fan_out"
    return "analyzer"


def build() -> StateGraph:
    wf = StateGraph(WorkflowState)
    wf.add_node("plan", boss_plan)
    wf.add_node("analyzer", run_analyzer)
    wf.add_node("review_analyzer", review_analyzer)
    wf.add_node("parallel_agents", run_parallel)
    wf.add_node("combine", boss_combine)

    wf.set_entry_point("plan")
    wf.add_edge("plan", "analyzer")
    wf.add_edge("analyzer", "review_analyzer")
    wf.add_conditional_edges(
        "review_analyzer", _route_analyzer,
        {"analyzer": "analyzer", "fan_out": "parallel_agents"},
    )
    wf.add_edge("parallel_agents", "combine")
    wf.add_edge("combine", END)

    return wf.compile()
