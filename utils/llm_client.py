"""LLM client — unified interface for any OpenAI-compatible API."""

from __future__ import annotations
import json, logging, os, time
from typing import TypeVar
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def _get_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


def llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    temperature: float = 0.2,
    max_retries: int = 3,
) -> T:
    client = _get_client()
    model = get_model()
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nRespond with JSON matching:\n{schema}\nOnly JSON — no markdown fences."},
        {"role": "user", "content": user_prompt},
    ]
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=4096, response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            return response_model.model_validate_json(raw)
        except Exception as exc:
            last_exc = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
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
        fb = f"\n\n--- PREVIOUS OUTPUT ---\n{previous_output}\n--- REVIEWER FEEDBACK ---\n{feedback}\n\nRewrite fixing ALL issues. Keep the same JSON format."
    return llm_call(system_prompt, user_prompt + fb, response_model, temperature=0.3, **kw)
