#!/usr/bin/env python3
"""CLI runner — analyze a paper from URL or local PDF."""

import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.pdf_parser import extract_text, fetch_from_url
from workflow.graph import build
from workflow.state import initial_state


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python run.py <arxiv-url-or-pdf-path> [--json]")
        return

    arg = sys.argv[1]
    as_json = "--json" in sys.argv

    if arg.startswith("http"):
        print(f"🌐 Fetching {arg}...")
        pdf = fetch_from_url(arg)
        text = extract_text(pdf)
    else:
        text = extract_text(arg)

    print(f"📄 Extracted {len(text):,} chars")
    print("🚀 Running agents...\n")

    graph = build()
    state = initial_state(paper_text=text)

    for event in graph.stream(state):
        for node, update in event.items():
            stage = update.get("stage", node)
            scores = update.get("review_scores", {})
            report = update.get("final_report")
            if scores:
                print(f"  [{stage}] Scores: {scores}")
            else:
                print(f"  [{stage}]")

    result = graph.invoke(state)
    fr = result.get("final_report")

    if as_json and fr:
        print(json.dumps(fr.model_dump(), indent=2, default=str))
    elif fr:
        print("\n" + "=" * 70)
        print(fr.raw_markdown)
        print("=" * 70)
        print(f"\nScores: {result.get('review_scores', {})}")
    else:
        print("\n⚠️  No report generated — check your API key and internet connection.")


if __name__ == "__main__":
    main()
