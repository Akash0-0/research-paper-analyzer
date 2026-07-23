"""LLM client — unified LangChain interface for any OpenAI-compatible API."""

from __future__ import annotations
import logging
import os
import time
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


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


def llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    temperature: float = 0.2,
    max_retries: int = 3,
) -> T:
    client = _get_client(temperature).with_structured_output(response_model)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            result = client.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            if isinstance(result, response_model):
                return result
            return response_model.model_validate(result)
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
