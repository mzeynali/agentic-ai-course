# Module 3 - Solution sketches

## E1 - Hierarchical

```python
crew = Crew(
    agents=[researcher, scheduler, responder],
    tasks=[task_triage, task_schedule, task_draft],
    process=Process.hierarchical,
    manager_llm=DEFAULT_MODEL,
)
```

Tradeoff: more flexible (the manager can reorder tasks) but less predictable
and more expensive. Use when the workflow genuinely varies per input.

## E2 - Sentiment agent

```python
sentiment = Agent(
    role="Tone Analyst",
    goal="Classify email sentiment as friendly/neutral/tense",
    backstory="You detect subtext.",
)
task_sentiment = Task(
    description="For each flagged email, classify tone.",
    expected_output="email_id -> tone mapping",
    agent=sentiment,
    context=[task_triage],
)
# Then add task_sentiment to the crew before task_schedule.
```

## E3 - Structured output

```python
from pydantic import BaseModel
from typing import List

class FlaggedEmail(BaseModel):
    email_id: str
    from_: str
    subject: str
    crm_context: str
    needs_meeting: bool

class TriageResult(BaseModel):
    emails: List[FlaggedEmail]

task_triage = Task(
    description=...,
    expected_output="...",
    agent=researcher,
    output_json=TriageResult,
)
```

## E4 - Retries

```python
task_triage = Task(..., max_retry_limit=2)
```

CrewAI will re-run the task if the tool throws.

## E5 - Token accounting

```python
crew = Crew(...)
result = crew.kickoff()
print(crew.usage_metrics)
```

Compare to LangChain/LangSmith's reported tokens for the LangGraph version,
and the `usage` object returned by each call in the Agents SDK version.
