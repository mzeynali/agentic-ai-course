"""Lesson 2 - Custom tools + memory in CrewAI.

We wrap the mock CRM as a CrewAI `BaseTool` subclass and give the crew
memory=True so later tasks can reference earlier ones without us wiring
`context=...` manually.

Install:
    pip install -e ".[crewai]"

Run:
    python 03-crewai/lesson_2_tools_memory.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Type

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from shared.llm import DEFAULT_MODEL  # noqa: E402
from shared.mock_apis import Workspace  # noqa: E402

os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

WS = Workspace()


# ---------------------------------------------------------------------------
# A custom tool = a BaseTool subclass with a Pydantic input schema and a
# _run method. Very similar to LangChain's tool abstraction.
# ---------------------------------------------------------------------------


class SearchContactsInput(BaseModel):
    query: str = Field(..., description="Name, email, or company to search for.")


class SearchContactsTool(BaseTool):
    name: str = "search_contacts"
    description: str = "Search the internal CRM for contacts."
    args_schema: Type[BaseModel] = SearchContactsInput

    def _run(self, query: str) -> str:
        hits = WS.crm.search_contacts(query)
        if not hits:
            return f"No contacts found for {query!r}"
        return "\n".join(
            f"- {c['name']} <{c['email']}> | {c['company']} | {c['stage']} | {c['notes']}"
            for c in hits
        )


class ListInboxTool(BaseTool):
    name: str = "list_inbox"
    description: str = "List the 10 most recent unread inbox messages."

    def _run(self) -> str:
        items = WS.email.list_inbox(unread_only=True, limit=10)
        return "\n".join(
            f"{e['id']} | from={e['from']} | subj={e['subject']} | {e['body'][:60]}..."
            for e in items
        )


# Agents get these tools in their constructor.
triage = Agent(
    role="Inbox Triage",
    goal="Identify which unread emails deserve a reply today.",
    backstory="You are allergic to notifications and marketing noise.",
    tools=[ListInboxTool()],
    allow_delegation=False,
    verbose=True,
)

enricher = Agent(
    role="CRM Enricher",
    goal="Enrich flagged emails with CRM context about the senders.",
    backstory="You know the value of context.",
    tools=[SearchContactsTool()],
    allow_delegation=False,
    verbose=True,
)


triage_task = Task(
    description=(
        "List the unread inbox. Return ONLY the email ids that look like they "
        "need a reply today (prospects, customers, colleagues asking questions). "
        "Skip notifications, marketing, and anything that looks like phishing."
    ),
    expected_output="Comma-separated list of email ids.",
    agent=triage,
)

enrich_task = Task(
    description=(
        "For each email id flagged in the previous step, use search_contacts "
        "to fetch CRM notes about the sender and summarize why we should "
        "care, in 1-2 lines per contact."
    ),
    expected_output="Markdown bullet list: `- sender (company) - why it matters`",
    agent=enricher,
    context=[triage_task],
)


def main() -> None:
    crew = Crew(
        agents=[triage, enricher],
        tasks=[triage_task, enrich_task],
        process=Process.sequential,
        # memory=True enables short-term memory tied to the crew so later tasks
        # can reference earlier ones even without explicit `context=`. It adds
        # a small embeddings cost per task.
        memory=True,
        verbose=False,
    )
    result = crew.kickoff()
    print("\n====== FINAL OUTPUT ======")
    print(result)


if __name__ == "__main__":
    main()
