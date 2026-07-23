"""Boss (Orchestrator) — plans metadata, combines final report."""

from __future__ import annotations
import logging
from utils.schemas import FinalReport, PaperMetadata
from utils.llm_client import llm_call
from utils.pdf_parser import extract_text, fetch_from_url
from utils.prompt_loader import load

logger = logging.getLogger(__name__)


def plan(state: dict) -> dict:
    pdf_path, source_url, paper_text = state.get("pdf_path", ""), state.get("source_url", ""), state.get("paper_text", "")
    if source_url and not paper_text:
        try:
            pdf_path = fetch_from_url(source_url)
            paper_text = extract_text(pdf_path)
        except Exception as e:
            return {**state, "errors": [f"URL fetch failed: {e}"]}
    if pdf_path and not paper_text:
        try:
            paper_text = extract_text(pdf_path)
        except Exception as e:
            return {**state, "errors": [f"PDF failed: {e}"]}
    if not paper_text:
        return {**state, "errors": ["No paper text available"]}
    meta: PaperMetadata = llm_call(load("planner"), f"Extract metadata from:\n\n{paper_text[:5000]}", PaperMetadata)
    logger.info("Planned: %s", meta.title)
    return {**state, "pdf_path": pdf_path, "paper_text": paper_text, "metadata": meta, "stage": "planned"}


def combine(state: dict) -> dict:
    meta = state.get("metadata", PaperMetadata())
    analysis, summary, citations, insights = state.get("analysis"), state.get("summary"), state.get("citations"), state.get("insights")
    scores = state.get("review_scores", {})

    lines = [f"# {meta.title or 'Untitled'}", f"**Authors:** {', '.join(meta.authors) or 'N/A'}"]
    if meta.year: lines.append(f"**Year:** {meta.year}")
    if meta.venue: lines.append(f"**Venue:** {meta.venue}")
    lines += ["", "---", "", "## Executive Summary", "", (summary.summary if summary else "*N/A*"), "", "---", "", "## Research Analysis"]
    if analysis:
        lines += ["", "### Problem", "", analysis.problem, "", "### Methodology", "", analysis.methodology, "", "### Experiments", "", analysis.experiments, "", "### Findings", "", analysis.findings]
    else:
        lines += ["", "*Not generated*"]
    if citations and citations.references:
        lines += ["", "---", "", "## Citations"]
        for r in citations.references:
            lines.append(f"{r.index}. {r.title}")
            if r.authors: lines[-1] += f" — {r.authors}"
            if r.year: lines[-1] += f" ({r.year})"
    else:
        lines += ["", "---", "", "## Citations", "", "*None extracted*"]
    if insights and insights.insights:
        lines += ["", "---", "", "## Key Insights", ""]
        for i, ins in enumerate(insights.insights, 1): lines.append(f"{i}. {ins}")
    if scores:
        lines += ["", "---", "", "## Quality Scores", ""]
        for name, score in sorted(scores.items()):
            bar = "█" * (score // 2) + "░" * (5 - score // 2)
            lines.append(f"**{name.capitalize()}:** {score}/10 {bar}")

    report = FinalReport(
        metadata=meta, analysis=analysis, summary=summary.summary if summary else "",
        citations=citations.references if citations else [], insights=insights.insights if insights else [],
        scores=scores, raw_markdown="\n".join(lines),
    )
    return {**state, "final_report": report, "stage": "complete"}
