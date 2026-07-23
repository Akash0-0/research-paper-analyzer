---
title: 🔬 Research Paper Analyzer
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🔬 AI-Powered Research Paper Analyzer

A **multi-agent system** that automatically reads, analyzes, and summarizes academic research papers. Built with **LangGraph** for stateful workflow orchestration and **Streamlit** for the UI.

## Architecture

```
User Uploads PDF / URL
         │
         ▼
  Planner / Boss        ← Extracts metadata
         │
         ▼
  Paper Analyzer        ← Problem, methodology, experiments, findings
         │
         ▼
  Reviewer              ← Quality gate (score ≥ 7 to pass)
         │
    ┌────┴────┐
    │ Score<7  │── retry with feedback (max 2x)
    ▼
  Summary │ Citation │ Insights    ← Parallel execution
         │
  Reviewer (each) → approve or retry
         │
  Boss (Combiner)        ← Final Markdown + JSON report
```

### Key Design Decisions

- **Boss never generates content** — only plans, delegates, and combines (real manager pattern)
- **Every agent returns Pydantic-validated JSON** — no free-form contamination
- **Provider-agnostic LLM** — works with OpenAI, any OpenAI-compat, or local models
- **Review gate with max 2 retries** — quality loop with graceful degradation
- **Parallel fan-out** after analyzer approval for Summary/Citation/Insights

## Project Structure

```
research-analyzer/
├── utils/          # schemas, LLM client, PDF parser, prompt loader
├── prompts/        # system prompts for each agent
├── workflow/       # LangGraph state, graph definition, agent nodes
├── app.py          # Streamlit UI
├── run.py          # CLI: python run.py <url-or-pdf>
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your API key
# OpenAI
export OPENAI_API_KEY="sk-..."
# Or any OpenAI-compatible provider
export LLM_API_KEY="..."
export LLM_BASE_URL="https://api.example.com/v1"
export LLM_MODEL="gpt-4o-mini"

# 3. Run
streamlit run app.py
# Or CLI:
python run.py https://arxiv.org/pdf/1706.03762.pdf
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | API key for your LLM provider | — |
| `LLM_BASE_URL` | Base URL for OpenAI-compatible API | OpenAI default |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |

## Sample Output

The system generates a comprehensive report including:

- **Paper Metadata** — title, authors, year, venue
- **Executive Summary** — 150-200 word overview
- **Research Analysis** — problem, methodology, experiments, findings
- **Citations** — all extracted references
- **Key Insights** — 3-5 actionable takeaways
- **Quality Scores** — per-agent review scores (1-10)

## Deployment

This Space runs on Hugging Face Spaces (Streamlit). The app is also fully functional as a local CLI tool.
