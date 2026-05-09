"""Wire Langfuse tracing into the LangGraph v1 agent.

Langfuse is a free open-source LLM observability platform. This file shows
the minimum you need to see every node, tool call, and token cost in the
Langfuse dashboard.

Setup (one-time, ~3 minutes):
  1. Go to https://cloud.langfuse.com and create a free account + project.
  2. Copy the Public Key and Secret Key into .env as:
        LANGFUSE_PUBLIC_KEY=pk-lf-...
        LANGFUSE_SECRET_KEY=sk-lf-...
        LANGFUSE_HOST=https://cloud.langfuse.com
  3. pip install -e ".[langgraph,production]"

Run:
    python 04-production/tracing_langfuse.py

Then open https://cloud.langfuse.com/project/<id>/traces to see your run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-langgraph"))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from project_workflow_v1 import SYSTEM_PROMPT, build_graph  # noqa: E402

load_dotenv()


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"{name} is not set. See the top of this file for Langfuse setup."
        )
    return v


def main() -> None:
    _require_env("LANGFUSE_PUBLIC_KEY")
    _require_env("LANGFUSE_SECRET_KEY")

    # Import late so the dependency check happens first.
    from langfuse.callback import CallbackHandler

    handler = CallbackHandler(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    graph = build_graph()
    config = {
        "configurable": {"thread_id": "langfuse-demo"},
        "callbacks": [handler],  # <-- this is the entire integration
        "metadata": {"course_module": "04-production", "version": "langgraph-v1"},
    }

    user_goal = (
        "Triage my unread inbox but do not send anything. Tell me which emails "
        "need a reply and why."
    )
    initial = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_goal),
        ]
    }
    # invoke() (not stream) so Langfuse sees clean parent/child spans.
    graph.invoke(initial, config=config)
    print("Run complete. Check your Langfuse dashboard for the trace.")


if __name__ == "__main__":
    main()
