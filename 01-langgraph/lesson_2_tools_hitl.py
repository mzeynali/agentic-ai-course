"""Lesson 2 - Tools with mock APIs + Human-in-the-loop approval.

This is the pattern you will use in production: the agent can do anything
safe autonomously (read email, search CRM), but any write action
(send_draft, create_event) pauses and waits for human approval.

Two important LangGraph primitives show up here:
  - `prebuilt.ToolNode`      : saves you writing the run_tools node by hand.
  - `interrupt()`            : suspends the graph until a human resumes it
                               with `Command(resume=...)`. Requires a
                               checkpointer, because "suspending" means saving
                               state to disk.

Run:
    python 01-langgraph/lesson_2_tools_hitl.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from shared.llm import DEFAULT_MODEL  # noqa: E402
from shared.mock_apis import Workspace  # noqa: E402

WS = Workspace()


class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Tools. Note how send_draft calls interrupt() - that's the human gate.
# Everything else executes normally.
# ---------------------------------------------------------------------------


@tool
def list_inbox(unread_only: bool = True, limit: int = 10) -> list[dict]:
    """List unread inbox messages."""
    return WS.email.list_inbox(unread_only=unread_only, limit=limit)


@tool
def get_email(email_id: str) -> dict:
    """Fetch full body of an email by id."""
    return WS.email.get_email(email_id)


@tool
def draft_reply(email_id: str, body: str, subject: str | None = None) -> str:
    """Create a draft reply. Returns the draft_id. Does NOT send."""
    return WS.email.draft_reply(email_id, body, subject)


@tool
def send_draft(draft_id: str) -> dict:
    """Send a previously-created draft. THIS REQUIRES HUMAN APPROVAL.

    The agent proposes; interrupt() pauses the run and surfaces the proposal
    to the caller; the caller resumes with 'approve' or 'deny'.
    """
    drafts = {d["id"]: d for d in WS.email.drafts()}
    draft = drafts.get(draft_id)
    if not draft:
        return {"error": f"draft {draft_id} not found"}

    decision = interrupt(
        {
            "action": "send_draft",
            "draft_id": draft_id,
            "preview": {
                "to": draft["to"],
                "subject": draft["subject"],
                "body": draft["body"],
            },
            "prompt": "Approve sending this email? Reply 'approve' or 'deny'.",
        }
    )

    if str(decision).strip().lower() not in {"approve", "yes", "y"}:
        return {"status": "denied_by_human", "draft_id": draft_id}
    return WS.email.send_draft(draft_id)


TOOLS = [list_inbox, get_email, draft_reply, send_draft]
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1).bind_tools(TOOLS)


def call_llm(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


def build_graph():
    g = StateGraph(State)
    g.add_node("llm", call_llm)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    # Checkpointer is required for interrupt() to work - state needs somewhere
    # to live while we wait for the human.
    return g.compile(checkpointer=MemorySaver())


SYSTEM_PROMPT = """You are a personal ops assistant. You can read email and
draft/send replies. Never send a draft without first showing a preview; the
system will automatically pause for human approval before sending."""


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    user_goal = (
        "Read my inbox, find Sara Chen's intro-call email, and draft a short "
        "polite reply proposing a 30-min call on Tuesday afternoon. Then send it."
    )
    print(f"USER: {user_goal}\n")

    # Run until the graph either finishes or hits an interrupt.
    initial = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_goal),
        ]
    }
    events = graph.stream(initial, config=config, stream_mode="updates")
    _drain_events(events)

    # Check whether we paused on an interrupt.
    state = graph.get_state(config)
    while state.interrupts:
        interrupt_obj = state.interrupts[0]
        payload = interrupt_obj.value
        print("\n==== HUMAN APPROVAL REQUESTED ====")
        print(f"Action   : {payload['action']}")
        preview = payload["preview"]
        print(f"To       : {preview['to']}")
        print(f"Subject  : {preview['subject']}")
        print(f"Body     :\n{preview['body']}")
        print("===================================")
        choice = input("Approve? [y/N] ").strip().lower() or "n"
        resume_val = "approve" if choice in {"y", "yes", "approve"} else "deny"
        events = graph.stream(
            Command(resume=resume_val), config=config, stream_mode="updates"
        )
        _drain_events(events)
        state = graph.get_state(config)

    print("\n-- sent messages --")
    for m in WS.email.sent_messages():
        print(f"  to={m['to']} subj={m['subject']!r}")


def _drain_events(events) -> None:
    for update in events:
        for node_name, node_update in update.items():
            msgs = node_update.get("messages", []) if isinstance(node_update, dict) else []
            for m in msgs:
                if isinstance(m, AIMessage) and m.tool_calls:
                    for c in m.tool_calls:
                        print(f"[{node_name}] tool: {c['name']}({c['args']})")
                elif isinstance(m, AIMessage):
                    print(f"[{node_name}] FINAL: {m.content}")


if __name__ == "__main__":
    main()
