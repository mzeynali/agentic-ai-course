"""Thin OpenAI wrapper used by every lesson.

Why a wrapper at all?
  - So lessons don't have to repeat the env-loading boilerplate.
  - So you can swap providers later (OpenRouter, local vLLM, Azure) by editing
    exactly one file instead of 20.
  - To centralize the "use the cheap model by default" policy.

Everything here is deliberately minimal. Frameworks (LangChain, OpenAI Agents
SDK, CrewAI) use their own clients internally - this module is only for the raw
calls in Module 0 and helper utilities.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a cached OpenAI client built from env vars.

    Honors OPENAI_API_KEY and optionally OPENAI_BASE_URL so you can point at
    OpenRouter / a local proxy without code changes.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key."
        )
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
    temperature: float = 0.2,
    **kwargs: Any,
):
    """One-shot chat completion with sensible defaults.

    Returns the raw `ChatCompletion` object so lessons can inspect tool_calls.
    """
    client = get_client()
    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    payload.update(kwargs)
    return client.chat.completions.create(**payload)
