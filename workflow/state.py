"""Single TypedDict carries everything through the graph."""

from __future__ import annotations
from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator
from utils.schemas import Analysis, CitationOutput, FinalReport, InsightOutput, PaperMetadata, ReviewScore, Summary


class WorkflowState(TypedDict):
    pdf_path: str
    paper_text: str
    source_url: str
    metadata: PaperMetadata
    analysis: Optional[Analysis]
    summary: Optional[Summary]
    citations: Optional[CitationOutput]
    insights: Optional[InsightOutput]
    review_scores: dict[str, int]
    review_feedback: dict[str, str]
    retry_counts: dict[str, int]
    analysis_approved: bool
    final_report: Optional[FinalReport]
    errors: Annotated[list[str], operator.add]
    stage: str  # human-readable stage name


def initial_state(pdf_path: str = "", paper_text: str = "", source_url: str = "") -> WorkflowState:
    return {
        "pdf_path": pdf_path,
        "paper_text": paper_text,
        "source_url": source_url,
        "metadata": PaperMetadata(),
        "analysis": None,
        "summary": None,
        "citations": None,
        "insights": None,
        "review_scores": {},
        "review_feedback": {},
        "retry_counts": {},
        "analysis_approved": False,
        "final_report": None,
        "errors": [],
        "stage": "starting",
    }
