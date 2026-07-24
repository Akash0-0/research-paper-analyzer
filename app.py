"""
Streamlit UI for the AI Research Paper Analyzer.
Upload a PDF or paste a URL, watch agents work live, get a full report.
"""

from __future__ import annotations
import logging, os, tempfile, threading, time

import streamlit as st

from utils.pdf_parser import extract_text, fetch_from_url
from workflow.graph import build
from workflow.state import initial_state

logging.basicConfig(level=logging.INFO)
# Suppress noisy HTTP request logging from the LLM client
for lib in ("httpx", "httpcore", "openai._base_client"):
    logging.getLogger(lib).setLevel(logging.WARNING)


# ── Helpers ────────────────────────────────────────────────────────────

def _render_scores(placeholder, scores: dict[str, int]):
    """Render score bars as custom HTML."""
    if not scores:
        return
    html = '<div class="score-card"><h4 style="margin: 0 0 0.5rem 0;">🏆 Quality Scores</h4>'
    for name in ["analyzer", "summary", "citation", "insights"]:
        s = scores.get(name, 0)
        bar = int(s / 10 * 100)
        html += f"""
        <div class="score-row">
            <div class="score-label">{name.capitalize()}</div>
            <div class="score-bar"><div class="score-fill" style="width:{bar}%"></div></div>
            <div class="score-value">{s}/10</div>
        </div>
        """
    html += "</div>"
    placeholder.markdown(html, unsafe_allow_html=True)


# ── Background thread runner ───────────────────────────────────────────

class WorkflowRunner:
    """Runs the graph in a background thread. UI polls for updates."""

    def __init__(self):
        self.stage: str = "Starting..."
        self.scores: dict[str, int] = {}
        self.report_md: str = ""
        self.done: bool = False
        self.error: str = ""

    def run(self, url: str, file_path: str | None):
        try:
            paper_text = ""
            pdf_path = ""
            source_url = ""

            if file_path and os.path.exists(file_path):
                self.stage = "📄 Extracting PDF text..."
                paper_text = extract_text(file_path)
                pdf_path = file_path
            elif url:
                self.stage = "🌐 Downloading PDF..."
                pdf_path = fetch_from_url(url)
                self.stage = "📄 Extracting text..."
                paper_text = extract_text(pdf_path)
                source_url = url
            else:
                self.error = "No input provided"
                self.done = True
                return

            if not paper_text.strip():
                self.error = "No text could be extracted from the PDF"
                self.done = True
                return

            self.stage = "⚙️ Building agent workflow..."
            graph = build()
            state = initial_state(
                pdf_path=pdf_path, paper_text=paper_text, source_url=source_url
            )

            self.stage = "🚀 Running agents..."
            for event in graph.stream(state):
                for node, update in event.items():
                    stage = update.get("stage", node)
                    scores = update.get("review_scores", {})
                    if scores:
                        self.scores = scores
                    if stage:
                        self.stage = stage
                    report = update.get("final_report")
                    if report:
                        self.report_md = report.raw_markdown

            result = graph.invoke(state)
            fr = result.get("final_report")
            if fr:
                self.report_md = fr.raw_markdown
                self.scores = result.get("review_scores", {})
                self.stage = "✅ Complete"
            else:
                self.stage = "⚠️ No report — check your API key and internet"
            self.done = True

        except Exception as e:
            self.stage = "❌ Error"
            self.error = str(e)
            self.done = True


# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.25rem; }
    .subheader { color: #666; margin-bottom: 2rem; }
    .score-card { background: #f8f9fa; border-radius: 12px; padding: 1rem 1.25rem; border: 1px solid #e9ecef; }
    .score-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0; }
    .score-label { width: 100px; font-weight: 600; font-size: 0.9rem; }
    .score-bar { flex: 1; height: 10px; background: #e9ecef; border-radius: 5px; overflow: hidden; }
    .score-fill { height: 100%; border-radius: 5px; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.3s ease; }
    .score-value { width: 40px; text-align: right; font-weight: 700; font-size: 0.9rem; }
    .stage-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────

if "runner" not in st.session_state:
    st.session_state.runner = None

# ── Title ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🔬 Research Paper Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">Multi-agent LangGraph system — reads, analyzes, and summarizes academic papers</div>',
    unsafe_allow_html=True,
)

# ── Input area ────────────────────────────────────────────────────────

url_input = st.text_input(
    "Paper URL",
    placeholder="https://arxiv.org/pdf/1706.03762.pdf",
    label_visibility="collapsed",
)
file_input = st.file_uploader("Upload PDF", type=["pdf"])

analyze = st.button("🔍 Analyze Paper", type="primary", use_container_width=True)

# ── Handle new analysis trigger ───────────────────────────────────────

if analyze:
    tmp_path = None
    if file_input is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(file_input.getvalue())
        tmp.close()
        tmp_path = tmp.name

    runner = WorkflowRunner()
    st.session_state.runner = runner
    thread = threading.Thread(target=runner.run, args=(url_input, tmp_path), daemon=True)
    thread.start()
    st.rerun()

# ── Status and results ────────────────────────────────────────────────

status_placeholder = st.empty()
scores_placeholder = st.empty()
report_placeholder = st.empty()

runner = st.session_state.runner

if runner is not None and not runner.done:
    # Polling — still running
    with status_placeholder.container():
        st.markdown(
            f'<div class="stage-badge" style="background:#e8f5e9;color:#2e7d32;">⏳ {runner.stage}</div>',
            unsafe_allow_html=True,
        )
        st.progress(0.5, text=runner.stage)

    if runner.scores:
        _render_scores(scores_placeholder, runner.scores)

    time.sleep(0.5)
    st.rerun()

elif runner is not None and runner.done:
    # Done — show final output
    with status_placeholder.container():
        if runner.error:
            st.error(f"❌ {runner.error}")
        elif runner.stage == "✅ Complete":
            st.success("✅ Analysis complete!")
        else:
            st.warning(runner.stage)

    if runner.scores:
        _render_scores(scores_placeholder, runner.scores)

    if runner.report_md:
        report_placeholder.markdown(runner.report_md)

    if st.button("🔄 New Analysis"):
        st.session_state.runner = None
        st.rerun()

elif runner is None:
    status_placeholder.info("Submit a paper URL or upload a PDF to begin analysis.")
