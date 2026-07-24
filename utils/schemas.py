"""Pydantic models for structured agent outputs — every agent returns JSON."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str = Field(default="", description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of author names — output as a JSON array")
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
    title: str = Field(default="", description="Paper title of this reference")
    authors: str = Field(default="", description="Author names as comma-separated string")
    year: Optional[int] = None
    venue: str = Field(default="", description="Conference or journal name")
    url: str = Field(default="")


class CitationOutput(BaseModel):
    references: list[Citation] = Field(default_factory=list, description="Array of citation objects. Each has: index (number), title (string), authors (string), year (number or null), venue (string), url (string)")


class InsightOutput(BaseModel):
    insights: list[str] = Field(default_factory=list, description="3-5 actionable takeaways")


class ReviewScore(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Quality score 1-10")
    feedback: str = Field(..., description="Detailed feedback")
    issues: list[str] = Field(default_factory=list, description="List of specific issues — output as a JSON array of strings")


class FinalReport(BaseModel):
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    analysis: Analysis | None = None
    summary: str = Field(default="")
    citations: list[Citation] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    scores: dict[str, int] = Field(default_factory=dict)
    raw_markdown: str = Field(default="", description="Full markdown report")
