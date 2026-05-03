"""Advanced ReAct agent for 00-setup — practical patterns beyond the basics.

New concepts demonstrated (each labelled in code):
  A. Structured tool results with error handling
  B. Memory: separate short-term (scratchpad) vs long-term (key-value store)
  C. Multi-step planning with explicit Thought / Action / Observation trace
  D. Tool retries with exponential back-off
  E. Token-budget guard (truncate old observations to avoid context overflow)
  F. Confidence gating — agent declares confidence before final answer
  G. Streaming the final answer to stdout character-by-character

Run:
    python 00-setup/advanced_agent.py
    python 00-setup/advanced_agent.py "What is the weather in Paris and Berlin? Save both to memory then summarise."
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.llm import get_client  # noqa: E402

# ---------------------------------------------------------------------------
# A. Structured tool results
#    Every tool returns {"ok": bool, "data": ..., "error": str | None}
#    This gives the model consistent signal about whether a call succeeded.
# ---------------------------------------------------------------------------

def _ok(data: Any) -> dict:
    return {"ok": True, "data": data, "error": None}


def _err(msg: str) -> dict:
    return {"ok": False, "data": None, "error": msg}


# ---------------------------------------------------------------------------
# B. Long-term memory — a simple in-process key-value store.
#    In production this would be a vector DB or Redis; here it's a dict.
# ---------------------------------------------------------------------------

MEMORY: dict[str, str] = {}


def memory_save(key: str, value: str) -> dict:
    MEMORY[key] = value
    return _ok(f"Saved '{key}'.")


def memory_load(key: str) -> dict:
    if key not in MEMORY:
        return _err(f"No entry for key '{key}'.")
    return _ok(MEMORY[key])


def memory_list_keys() -> dict:
    return _ok(list(MEMORY.keys()))


# ---------------------------------------------------------------------------
# Existing tools — now return structured dicts
# ---------------------------------------------------------------------------

FAKE_WEATHER = {
    "san francisco": "62°F, foggy",
    "tehran": "78°F, sunny",
    "paris": "65°F, partly cloudy",
    "berlin": "55°F, overcast",
    "tokyo": "72°F, humid",
}


def get_weather(city: str) -> dict:
    key = city.strip().lower()
    if not key:
        return _err("City name cannot be empty.")
    result = FAKE_WEATHER.get(key, f"70°F, clear (simulated for {city})")
    return _ok(result)


def get_time(timezone: str = "UTC") -> dict:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return _ok(f"Current UTC time: {ts}  (requested tz: {timezone})")


def calculator(expression: str) -> dict:
    """Safely evaluate simple math expressions."""
    allowed = set("0123456789+-*/(). ")
    if any(c not in allowed for c in expression):
        return _err(f"Unsafe expression: '{expression}'")
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return _ok(result)
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Tool registry — maps name → (function, schema)
# ---------------------------------------------------------------------------

TOOL_IMPLS: dict[str, Any] = {
    "get_weather": get_weather,
    "get_time": get_time,
    "calculator": calculator,
    "memory_save": memory_save,
    "memory_load": memory_load,
    "memory_list_keys": memory_list_keys,
}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current (fake) weather for a real geographic city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Paris'."}
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
            "description": "Get the current UTC time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "default": "UTC"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple arithmetic expression like '(3+5)*2'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression string."}
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "Persist a value under a key for later retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_load",
            "description": "Retrieve a previously saved value by key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list_keys",
            "description": "List all keys currently stored in memory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# D. Tool call with retry + exponential back-off
# ---------------------------------------------------------------------------

def call_tool_with_retry(name: str, args: dict, max_retries: int = 2) -> str:
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return json.dumps(_err(f"Unknown tool: {name}"))

    for attempt in range(max_retries + 1):
        try:
            result = impl(**args)
            return json.dumps(result)
        except Exception as exc:
            if attempt == max_retries:
                return json.dumps(_err(f"Tool crashed after {max_retries+1} tries: {exc}"))
            wait = 2 ** attempt
            print(f"     [retry {attempt+1}/{max_retries}] sleeping {wait}s after error: {exc}")
            time.sleep(wait)

    return json.dumps(_err("Unexpected retry exhaustion"))


# ---------------------------------------------------------------------------
# E. Token-budget guard
#    Keep only the system message + last N tool observations to prevent
#    unbounded context growth on long tasks.
# ---------------------------------------------------------------------------

MAX_OBSERVATION_MESSAGES = 10


def trim_messages(messages: list[dict]) -> list[dict]:
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    tool_msgs = [m for m in rest if m.get("role") == "tool"]
    non_tool = [m for m in rest if m.get("role") != "tool"]

    if len(tool_msgs) > MAX_OBSERVATION_MESSAGES:
        dropped = len(tool_msgs) - MAX_OBSERVATION_MESSAGES
        print(f"  [context trim] dropped {dropped} old tool observations")
        tool_msgs = tool_msgs[-MAX_OBSERVATION_MESSAGES:]

    combined = non_tool + tool_msgs
    combined.sort(key=lambda m: rest.index(m) if m in rest else 9999)
    return system + combined


# ---------------------------------------------------------------------------
# F. System prompt: ask for explicit confidence before answering
#    G. Streaming is done in the final-answer block below.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise, concise assistant with access to tools.

## Reasoning protocol (ReAct style)
Before every tool call, write one short "Thought:" line explaining WHY you need it.
After receiving a tool result, write one short "Observation:" line summarising what you learned.

## Tool policy
- Call a tool only when it directly answers a sub-question.
- Factual / math questions you already know: answer from knowledge (use calculator for arithmetic).
- memory_save / memory_load: use these when the user explicitly asks to remember something.
- Never invent arguments a tool was not designed for.

## Confidence gate (F)
When you are ready to give the final answer, first output:
  Confidence: <HIGH|MEDIUM|LOW> — <one-sentence reason>
Then give the actual answer.

## Style
Be terse. Bullet-points over prose. No filler phrases.
"""


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(user_goal: str, max_steps: int = 10) -> str:
    client = get_client()
    import os
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]

    print(f"\n{'='*60}")
    print(f"USER: {user_goal}")
    print(f"{'='*60}")

    for step in range(max_steps):
        # E. trim context before each call
        messages = trim_messages(messages)

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # ---- No tool calls → final answer (with streaming simulation) --------
        if not msg.tool_calls:
            content = msg.content or ""

            # G. Stream final answer character-by-character
            print(f"\n[final answer @ step {step}]")
            for char in content:
                print(char, end="", flush=True)
                time.sleep(0.005)  # simulate streaming; remove in real usage
            print()
            return content

        # ---- Tool calls -------------------------------------------------------
        print(f"\n[step {step}] — {len(msg.tool_calls)} tool call(s):")
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"  -> {name}({json.dumps(args)})")

            # D. call with retry
            result_str = call_tool_with_retry(name, args)
            result_data = json.loads(result_str)

            status = "ok" if result_data.get("ok") else "ERR"
            print(f"     [{status}] {result_data.get('data') or result_data.get('error')}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": result_str,
                }
            )

    return "[agent] hit max_steps without a final answer"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    goal = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "What's the weather in Paris and Berlin? "
            "Save each result to memory (keys: 'weather_paris', 'weather_berlin'). "
            "Then calculate the average of 65 and 55. "
            "Finally, summarise everything."
        )
    )
    run_agent(goal)
