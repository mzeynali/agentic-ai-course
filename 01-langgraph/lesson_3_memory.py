"""Lesson 3 - Short-term and long-term memory in LangGraph.

Two kinds of memory:

  (1) Short-term (thread-scoped, per-conversation):
      Handled by a **checkpointer**. Every node update is saved under a
      `thread_id` so if the user says "what did I just ask?", the model can
      see the full prior transcript. Same mechanism that powers HITL pauses.

  (2) Long-term (cross-thread, per-user / per-namespace):
      Handled by a **store**. You write facts ("user prefers 30-min meetings",
      "Sara Chen is a prospect from Acme") that survive even if the thread is
      deleted. LangGraph nodes can read/write the store via the `config`
      argument or a `BaseStore` injected at compile time.

In this lesson we teach the agent to remember user preferences across
conversations by writing to an InMemoryStore, then start a SECOND thread and
see the preferences carried over.

Run:
    python 01-langgraph/lesson_3_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from shared.llm import DEFAULT_MODEL  # noqa: E402


# ---------------------------------------------------------------------------
# A minimal user-profile tool surface. The agent decides when to save and
# when to recall.
# ---------------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


@tool
def remember(fact: str) -> str:
    """Save a durable fact about the current user (e.g. a preference).

    The framework injects the store and user_id via `config`; the tool body
    is intentionally a no-op so the LLM calls it with clean semantics. The
    real work happens in the `_persist_memories` node below.
    """
    return f"ok - will persist: {fact}"


@tool
def recall(topic: str) -> str:
    """Look up previously saved facts about the user that match `topic`."""
    return f"ok - will look up: {topic}"


TOOLS = [remember, recall]
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0).bind_tools(TOOLS)


def call_llm(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    # Inject whatever we remember about this user as a system hint.
    user_id = state["user_id"]
    facts = store.search(("users", user_id, "facts"), query="preferences", limit=5)
    hint = "\n".join(f"- {f.value['fact']}" for f in facts) or "(no saved facts yet)"
    system = SystemMessage(
        content=(
            "You are an assistant with long-term memory.\n"
            "When the user shares a preference, call `remember`.\n"
            "When you need to personalize, call `recall`.\n"
            f"Known facts about this user:\n{hint}"
        )
    )
    return {"messages": [llm.invoke([system, *state["messages"]])]}


def _persist_memories(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    """After tool execution, intercept remember() calls and write to the store."""
    user_id = state["user_id"]
    # Walk backwards to find the most recent AIMessage with tool_calls.
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                if call["name"] == "remember":
                    fact = call["args"].get("fact", "").strip()
                    if fact:
                        store.put(
                            ("users", user_id, "facts"),
                            key=f"fact_{abs(hash(fact))}",
                            value={"fact": fact},
                        )
            break
    return {}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


def build_graph():
    g = StateGraph(State)
    g.add_node("llm", call_llm)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("persist", _persist_memories)
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "persist")
    g.add_edge("persist", "llm")
    return g.compile(checkpointer=MemorySaver(), store=InMemoryStore())


def main() -> None:
    graph = build_graph()

    # First conversation: teach the agent a preference.
    cfg1 = {"configurable": {"thread_id": "t-morning", "user_id": "me"}}
    print("\n=== THREAD 1: morning ===")
    for update in graph.stream(
        {
            "messages": [HumanMessage(content="Please remember: I prefer 25-minute meetings and hate meetings before 10am local time.")],
            "user_id": "me",
        },
        config=cfg1,
        stream_mode="values",
    ):
        last = update["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            print(f"agent: {last.content}")

    # Second conversation (different thread_id): the store persists, but the
    # checkpointer does not share state across threads.
    cfg2 = {"configurable": {"thread_id": "t-afternoon", "user_id": "me"}}
    print("\n=== THREAD 2: afternoon (new thread, same user) ===")
    for update in graph.stream(
        {
            "messages": [HumanMessage(content="How long should our next meeting be and when should I avoid it?")],
            "user_id": "me",
        },
        config=cfg2,
        stream_mode="values",
    ):
        last = update["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            print(f"agent: {last.content}")


if __name__ == "__main__":
    main()
