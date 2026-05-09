"""Lesson 1 - LangGraph basics: StateGraph, nodes, conditional edges.

We rebuild the Module-0 ReAct loop, but this time as an explicit graph. The
same two tools (weather + time) are used so you can diff the files and see
exactly what LangGraph adds.

Install:
    pip install -e ".[langgraph]"

Run:
    python 01-langgraph/lesson_1_graph_basics.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from shared.llm import DEFAULT_MODEL  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Define the state.
#    `add_messages` is a reducer - it merges new messages into the list rather
#    than replacing it. This is how LangGraph keeps a rolling conversation.
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 2. Define tools with @tool. LangChain auto-generates JSON schemas for them.
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {"san francisco": "62F and foggy", "tehran": "78F and sunny"}
    return fake.get(city.lower(), f"weather for {city}: 70F and clear (fake)")


@tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current UTC time."""
    return f"Current UTC time is {datetime.utcnow().isoformat(timespec='seconds')}Z"


TOOLS = [get_weather, get_time]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2).bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 3. Define the nodes. Each node takes state and returns a partial update.
# ---------------------------------------------------------------------------
def call_llm(state: AgentState) -> dict:
    """Ask the LLM what to do next given the conversation so far."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def run_tools(state: AgentState) -> dict:
    """Execute every tool_call in the last AIMessage and feed results back."""
    last = state["messages"][-1]
    assert isinstance(last, AIMessage) and last.tool_calls, "run_tools called with no pending tool_calls"

    tool_messages: list[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args", {})
        tool_obj = TOOLS_BY_NAME[name]
        try:
            result = tool_obj.invoke(args)
        except Exception as exc:  # noqa: BLE001
            result = f"ERROR running {name}: {exc!r}"
        tool_messages.append(
            ToolMessage(
                content=json.dumps(result) if not isinstance(result, str) else result,
                tool_call_id=call["id"],
                name=name,
            )
        )
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """Conditional edge: if the LLM asked for tools, run them; otherwise stop."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 4. Assemble the graph.
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("llm", call_llm)
    g.add_node("tools", run_tools)
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    return g.compile()


def main() -> None:
    graph = build_graph()
    user_goal = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What's the weather in Tehran and the current UTC time?"
    )
    initial = {
        "messages": [
            SystemMessage(content="You are a terse assistant. Use tools when needed."),
            HumanMessage(content=user_goal),
        ]
    }

    # .stream() yields one update per node execution - great for seeing the flow.
    print(f"USER: {user_goal}\n")
    for update in graph.stream(initial, stream_mode="updates"):
        for node_name, node_update in update.items():
            last = node_update["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                calls = ", ".join(f"{c['name']}({c['args']})" for c in last.tool_calls)
                print(f"[{node_name}] -> tool_calls: {calls}")
            elif isinstance(last, AIMessage):
                print(f"[{node_name}] -> FINAL: {last.content}")
            elif isinstance(last, ToolMessage):
                print(f"[{node_name}] -> {last.name} returned: {last.content}")


if __name__ == "__main__":
    main()
