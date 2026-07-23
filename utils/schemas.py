"""Pydantic models for structured agent outputs — every agent returns JSON."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str = Field(default="", description="Paper title")
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = Field(default="")


class Analysis(BaseModel):
    problem: str = Field(..., description="Problem statement / research gap")
    methodology: str = Field(..., description="Methodology and approach")
    experiments: str = Field(..., description="Key experiments and setup")
    findings: str = Field(..., description="Main findings and results")


class Summary(BaseModel):
    summary: str = Field(..., description="150-200 word executive summary")


class Citation(BaseModel):
    index: int = Field(..., description="Citation number")
    title: str = Field(default="")
    authors: str = Field(default="")
    year: Optional[int] = None
    venue: str = Field(default="")
    url: str = Field(default="")


class CitationOutput(BaseModel):
    references: list[Citation] = Field(default_factory=list)


class InsightOutput(BaseModel):
    insights: list[str] = Field(default_factory=list, description="3-5 actionable takeaways")


class ReviewScore(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Quality score 1-10")
    feedback: str = Field(..., description="Detailed feedback")
    issues: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    analysis: Analysis | None = None
    summary: str = Field(default="")
    citations: list[Citation] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    scores: dict[str, int] = Field(default_factory=dict)
    raw_markdown: str = Field(default="", description="Full markdown report")
