"""Lesson 1 - OpenAI Agents SDK basics: Agent, @function_tool, Runner.

Install:
    pip install -e ".[openai-agents]"

Run:
    python 02-openai-agents/lesson_1_agents_tools.py

Notice how much ceremony disappears compared to LangGraph L1: no graph, no
state dict, no conditional edges. The Runner internally does the
message/tool loop we wrote by hand in Module 0.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Agent, Runner, function_tool

from shared.llm import DEFAULT_MODEL  # noqa: E402


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {"san francisco": "62F and foggy", "tehran": "78F and sunny"}
    return fake.get(city.lower(), f"weather for {city}: 70F and clear (fake)")


@function_tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current UTC time."""
    return f"Current UTC time is {datetime.utcnow().isoformat(timespec='seconds')}Z"


def build_agent() -> Agent:
    return Agent(
        name="WeatherTimeAgent",
        instructions=(
            "You are a terse assistant. Use tools when needed. "
            "When you have enough information, respond with the final answer."
        ),
        tools=[get_weather, get_time],
        model=DEFAULT_MODEL,
    )


async def run_streaming(agent: Agent, prompt: str) -> None:
    """Stream events as they happen - best way to watch the loop live."""
    print(f"USER: {prompt}\n")
    result = Runner.run_streamed(agent, prompt)
    async for event in result.stream_events():
        # Event types include 'run_item_stream_event', 'raw_response_event', etc.
        # We'll only pretty-print the high-level run items.
        if event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                name = getattr(item.raw_item, "name", None) or "<tool>"
                args = getattr(item.raw_item, "arguments", "{}")
                print(f"[tool_call] {name}({args})")
            elif item.type == "tool_call_output_item":
                print(f"[tool_out ] {str(item.output)[:120]}")
            elif item.type == "message_output_item":
                print(f"[message  ] {result.final_output if result.final_output else '...'}")
    print(f"\nFINAL: {result.final_output}")


def main() -> None:
    agent = build_agent()
    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What's the weather in Tehran and the current UTC time?"
    )
    # Sync version - simpler, but you lose streaming.
    result = Runner.run_sync(agent, prompt)
    print(f"USER: {prompt}")
    print(f"FINAL (sync): {result.final_output}\n")

    # Async streaming version - same result, with live progress.
    asyncio.run(run_streaming(agent, prompt))


if __name__ == "__main__":
    main()
