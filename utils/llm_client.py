"""LLM client — unified interface for any OpenAI-compatible API."""

from __future__ import annotations
import json
import logging
import os
import re
import time
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """Strip surrounding text, markdown fences, and 'json' prefixes from LLM output."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Remove leading "json" or "JSON" on its own line
    raw = re.sub(r"^(?i:json)\s*\n", "", raw)
    raw = raw.strip()
    # Find the first { or [ and last } or ]
    start = -1
    for ch in ("{", "["):
        idx = raw.find(ch)
        if idx != -1 and (start == -1 or idx < start):
            start = idx
    end = -1
    for ch in ("}", "]"):
        idx = raw.rfind(ch)
        if idx != -1 and (end == -1 or idx > end):
            end = idx + 1
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end]
    return raw.strip()


def _get_client(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
    return ChatOpenAI(
        model=get_model(),
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=4096,
    )


def get_model() -> str:
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


# ── Core functions ─────────────────────────────────────────────────────

def llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    temperature: float = 0.2,
    max_retries: int = 3,
) -> T:
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    messages = [
        ("system", f"{system_prompt}\n\nReturn ONLY valid JSON matching this schema — no markdown, no 'json' prefix:\n{schema}"),
        ("human", user_prompt),
    ]

    client = _get_client(temperature)
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            result = client.invoke(messages)

            # LangChain with_structured_output returns the model directly
            if isinstance(result, response_model):
                return result

            # Raw string response — extract JSON and validate
            if hasattr(result, "content"):
                raw = result.content
            elif isinstance(result, str):
                raw = result
            else:
                raw = str(result)

            cleaned = _extract_json(raw)
            if not cleaned:
                raise ValueError(f"Empty response after extraction. Raw: {raw[:200]}")

            return response_model.model_validate_json(cleaned)

        except (ValidationError, ValueError, Exception) as exc:
            last_exc = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2**attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_exc}")


def llm_call_with_feedback(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    previous_output: str = "",
    feedback: str = "",
    **kw,
) -> T:
    fb = ""
    if previous_output and feedback:
        fb = (
            f"\n\n--- PREVIOUS OUTPUT ---\n{previous_output}"
            f"\n--- REVIEWER FEEDBACK ---\n{feedback}"
            f"\n\nRewrite fixing ALL issues. Only JSON."
        )
    return llm_call(system_prompt, user_prompt + fb, response_model, temperature=0.3, **kw)
