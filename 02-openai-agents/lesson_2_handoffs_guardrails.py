"""Lesson 2 - Handoffs + input guardrails.

Pattern: a Triage agent classifies the incoming request and hands off to a
specialist. Before doing anything, an input guardrail blocks obvious off-topic
or unsafe requests.

Install:
    pip install -e ".[openai-agents]"

Run:
    python 02-openai-agents/lesson_2_handoffs_guardrails.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    Runner,
    function_tool,
)
from pydantic import BaseModel

from shared.llm import DEFAULT_MODEL  # noqa: E402
from shared.mock_apis import Workspace  # noqa: E402

WS = Workspace()


# ---------- tools ----------------------------------------------------------


@function_tool
def list_inbox(unread_only: bool = True, limit: int = 10) -> list[dict]:
    """List unread inbox messages."""
    return WS.email.list_inbox(unread_only=unread_only, limit=limit)


@function_tool
def find_free_slots(start_iso: str, end_iso: str, duration_minutes: int = 30) -> list[dict]:
    """Find free calendar slots."""
    return WS.calendar.find_free_slots(start_iso, end_iso, duration_minutes)


@function_tool
def draft_reply(email_id: str, body: str, subject: str | None = None) -> str:
    """Draft (but do not send) a reply."""
    return WS.email.draft_reply(email_id, body, subject)


# ---------- specialists -----------------------------------------------------

scheduler = Agent(
    name="SchedulerAgent",
    handoff_description="Specialist for proposing meeting times from the calendar.",
    instructions=(
        "You propose meeting slots. Use find_free_slots once, then return 2-3 "
        "options to the user in ISO format. Never create events."
    ),
    tools=[find_free_slots],
    model=DEFAULT_MODEL,
)

responder = Agent(
    name="ResponderAgent",
    handoff_description="Specialist for drafting polite, concise email replies.",
    instructions=(
        "You draft replies. Use list_inbox or draft_reply as needed. Never "
        "send - only draft. Keep replies under 120 words."
    ),
    tools=[list_inbox, draft_reply],
    model=DEFAULT_MODEL,
)


# ---------- guardrail ------------------------------------------------------


class OnTopicCheck(BaseModel):
    on_topic: bool
    reason: str


topic_gate = Agent(
    name="TopicGate",
    instructions=(
        "Return on_topic=true ONLY if the user request is about email triage, "
        "calendar scheduling, contact lookup, or drafting replies. Otherwise "
        "on_topic=false."
    ),
    model=DEFAULT_MODEL,
    output_type=OnTopicCheck,
)


async def on_topic_guardrail(ctx, agent, input_data):
    result = await Runner.run(topic_gate, input_data, context=ctx.context)
    check = result.final_output_as(OnTopicCheck)
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.on_topic,
    )


# ---------- triage ---------------------------------------------------------

triage = Agent(
    name="TriageAgent",
    instructions=(
        "You are the triage for a personal-ops assistant. Inspect the user "
        "request and hand off to SchedulerAgent for time-slot questions, or to "
        "ResponderAgent for anything about email. If both are needed, start "
        "with the Responder and let it hand off."
    ),
    handoffs=[scheduler, responder],
    input_guardrails=[InputGuardrail(guardrail_function=on_topic_guardrail)],
    model=DEFAULT_MODEL,
)


def main() -> None:
    # 1. On-topic request: should succeed and hand off.
    ok_prompt = (
        "Find 2-3 open 30-minute slots next Tuesday or Wednesday afternoon "
        "for an intro call."
    )
    print(f"USER: {ok_prompt}")
    res = Runner.run_sync(triage, ok_prompt)
    print(f"FINAL: {res.final_output}\n")

    # 2. Off-topic request: guardrail should trip.
    bad_prompt = "Write me a poem about llamas."
    print(f"USER: {bad_prompt}")
    try:
        Runner.run_sync(triage, bad_prompt)
    except InputGuardrailTripwireTriggered as exc:
        print(f"GUARDRAIL TRIPPED (expected): {exc}")


if __name__ == "__main__":
    main()
