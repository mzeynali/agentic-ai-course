"""The smallest useful agent: a raw ReAct loop in ~50 lines of real logic.

Run with:
    python 00-setup/hello_agent.py

What to notice while reading:
  1. There is no framework. It's just a while-loop, a call to the OpenAI API,
     and a tiny dispatch table mapping tool names to Python functions.
  2. On every iteration the ENTIRE message history (including past tool
     results) is re-sent to the model. That's the "context" the model uses to
     decide what to do next.
  3. The loop ends when the model responds with no tool calls - that's the
     "final answer".
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `shared` imports work

from shared.llm import chat  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Define the tools.
#    Each tool is (a) a JSON schema the model sees, and (b) a Python function
#    that actually runs when the model emits a matching tool_call.
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    fake = {"san francisco": "62F and foggy", "tehran": "78F and sunny"}
    return fake.get(city.lower(), f"weather for {city}: 70F and clear (fake)")


def get_time(timezone: str = "UTC") -> str:
    return f"Current UTC time is {datetime.utcnow().isoformat(timespec='seconds')}Z"


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a real geographic city."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of a real city, e.g. 'Tehran', 'San Francisco'.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time, optionally in a given timezone.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "default": "UTC"}},
            },
        },
    },
]

TOOL_IMPLS = {"get_weather": get_weather, "get_time": get_time}


# ---------------------------------------------------------------------------
# 2. The loop itself. This is the entire "agent".
# ---------------------------------------------------------------------------

def run_agent(user_goal: str, max_steps: int = 6) -> str:
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a terse assistant.\n"
                "Tool-use policy:\n"
                "  - Answer ONLY what the user explicitly asked. Do NOT volunteer extra "
                "information or extra tool calls the user did not request.\n"
                "  - Call a tool ONLY when it directly answers a sub-question the user actually asked "
                "(get_weather is for current weather in a real geographic city; "
                "get_time is for the current time).\n"
                "  - If a sub-question is factual/general knowledge (capitals, history, definitions, "
                "math, etc.), answer it from your own knowledge. Do NOT call a tool for it.\n"
                "  - NEVER misuse a tool by passing arguments it wasn't designed for "
                "(e.g. do not call get_weather with a crypto ticker, stock symbol, or non-city string).\n"
                "  - If your knowledge may be out of date (live prices, current news), say so briefly.\n"
                "Examples:\n"
                "  - 'Weather in Tehran and capital of USA?' -> call get_weather('Tehran'); "
                "answer 'Washington, D.C.' from knowledge. Do NOT call get_weather on Washington.\n"
                "  - 'Weather in Paris and who wrote Hamlet?' -> call get_weather('Paris'); "
                "answer 'Shakespeare' from knowledge. No second tool call.\n"
                "When you have enough information, respond with the final answer."
            ),
        },
        {"role": "user", "content": user_goal},
    ]

    for step in range(max_steps):
        resp = chat(messages, tools=TOOLS_SPEC)
        msg = resp.choices[0].message

        # Persist the assistant turn (tool-calls included) so the model can see
        # its own prior reasoning on the next iteration.
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print(f"\n[final @ step {step}] {msg.content}")
            return msg.content or ""

        print(f"\n[step {step}] model wants to call {len(msg.tool_calls)} tool(s):")
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"  -> {name}({args})")
            impl = TOOL_IMPLS.get(name)
            result = impl(**args) if impl else f"ERROR: unknown tool {name}"
            print(f"     = {result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": str(result),
                }
            )

    return "[agent] hit max_steps without a final answer"


if __name__ == "__main__":
    goal = (
        sys.argv[1]
        if len(sys.argv) > 1
        # else "What's the weather in Tehran right now, and what time is it in UTC?"
        else "What's the weather in Tehran right now, and what's the capital of the USA?"
    )
    print(f"USER: {goal}")
    run_agent(goal)
