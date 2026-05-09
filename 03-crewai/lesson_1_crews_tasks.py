"""Lesson 1 - CrewAI basics: Agents, Tasks, Crews, Process.

Install:
    pip install -e ".[crewai]"

Run:
    python 03-crewai/lesson_1_crews_tasks.py

Notice how CrewAI encourages you to describe the *who* and the *what* in
plain language - closer to writing a job description than coding a control
flow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crewai import Agent, Crew, Process, Task

from shared.llm import DEFAULT_MODEL  # noqa: E402

# CrewAI reads OPENAI_MODEL_NAME to pick a default. Pin it to our cheap model
# so lessons don't accidentally burn tokens on gpt-4-turbo.
os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)


researcher = Agent(
    role="Research Analyst",
    goal="Produce a concise, factual brief on a given topic.",
    backstory=(
        "You are a senior research analyst who prefers short paragraphs to long "
        "ones and always cites the 2-3 most important facts."
    ),
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Turn raw research into a 3-bullet executive summary.",
    backstory="You distill briefs into at-a-glance bullets for busy readers.",
    allow_delegation=False,
    verbose=True,
)


research_task = Task(
    description=(
        "Write a 5-sentence brief on how LLM agents use tool calling. "
        "Mention: (a) the ReAct loop, (b) the JSON schema for tool definitions, "
        "(c) one failure mode."
    ),
    expected_output="A 5-sentence brief paragraph.",
    agent=researcher,
)

summary_task = Task(
    description="Convert the brief into exactly 3 bullet points.",
    expected_output="Three bullet points, each 1 sentence.",
    agent=writer,
    context=[research_task],  # writer sees researcher's output
)


def run_sequential() -> None:
    print("\n=== Process.sequential ===")
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, summary_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    print(f"\nFINAL:\n{result}")


def run_hierarchical() -> None:
    """With Process.hierarchical, a manager LLM decides which agent gets what
    and in what order. You don't need to pre-wire task dependencies.
    """
    print("\n=== Process.hierarchical ===")
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, summary_task],
        process=Process.hierarchical,
        manager_llm=DEFAULT_MODEL,
        verbose=False,
    )
    result = crew.kickoff()
    print(f"\nFINAL:\n{result}")


if __name__ == "__main__":
    run_sequential()
    # Uncomment to see the manager-coordinated version. It costs more but
    # demonstrates how hierarchical crews can improvise beyond a fixed plan.
    # run_hierarchical()
