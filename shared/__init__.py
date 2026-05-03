"""Shared utilities for every lesson in the course.

Exposes:
  - get_client(): configured OpenAI client (honors env vars)
  - DEFAULT_MODEL: the model most lessons use (gpt-4o-mini)
  - mock_apis: in-memory Email / Calendar / CRM SDKs
  - eval_harness: tiny LLM-as-judge runner
"""

from .llm import DEFAULT_MODEL, chat, get_client

__all__ = ["DEFAULT_MODEL", "chat", "get_client"]
