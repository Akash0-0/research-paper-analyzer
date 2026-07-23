"""PDF text extraction — tries pymupdf, falls back to pdfplumber then PyPDF2."""

from __future__ import annotations
import logging, re, tempfile
import requests

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> str:
    for lib_name, try_import, extract_fn in [
        ("pymupdf", lambda: __import__("pymupdf"), _extract_pymupdf),
        ("pdfplumber", lambda: __import__("pdfplumber"), _extract_plumber),
        ("PyPDF2", lambda: __import__("PyPDF2"), _extract_pypdf2),
    ]:
        try:
            try_import()
            text = extract_fn(pdf_path)
            if text.strip():
                logger.info("Extracted %d chars via %s", len(text), lib_name)
                return _clean(text)
        except Exception:
            continue
    raise RuntimeError("No PDF library available — install pymupdf, pdfplumber, or PyPDF2")


def fetch_from_url(url: str) -> str:
    import re
    arxiv = re.match(r"https?://arxiv\.org/abs/(\d+\.\d+)", url)
    if arxiv:
        url = f"https://arxiv.org/pdf/{arxiv.group(1)}.pdf"
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in resp.iter_content(8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def _extract_pymupdf(path: str) -> str:
    import pymupdf
    doc = pymupdf.open(path)
    text = "\n\n".join(p.get_text() for p in doc)
    doc.close()
    return text


def _extract_plumber(path: str) -> str:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(p.extract_text() or "" for p in pdf.pages)


def _extract_pypdf2(path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\x00", "")
    for a, b in [("\u2013", "-"), ("\u2014", "--"), ("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')]:
        text = text.replace(a, b)
    return text.strip()


def chunk_text(text: str, max_chars: int = 8_000, overlap: int = 500) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            pe = text.rfind("\n\n", start, end)
            if pe > start + max_chars // 2:
                end = pe + 1
            else:
                se = text.rfind(". ", start, end)
                if se > start + max_chars // 2:
                    end = se + 2
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
