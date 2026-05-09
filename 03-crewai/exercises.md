# Module 3 - Exercises

## E1 - Hierarchical mode

Flip `project_workflow_v3.py` from `Process.sequential` to
`Process.hierarchical`, set a `manager_llm=DEFAULT_MODEL`, and remove the
`context=[...]` wiring from the tasks. Re-run. Does the manager still produce
correct results? What went wrong (or right)?

## E2 - Add a Sentiment agent

Add a new `SentimentAgent` between the Researcher and the Scheduler that
flags each flagged email as `{friendly, neutral, tense}`. The Responder
should adjust tone accordingly in its drafts.

## E3 - Structured task output

CrewAI tasks accept `output_json=SomeModel` (Pydantic) so you get typed
results instead of raw strings. Convert `task_triage` to emit a
`list[FlaggedEmail]` where `FlaggedEmail` is a Pydantic model.

## E4 - Custom tool error handling

Edit `ListInboxTool._run` to raise randomly (e.g. 30% of the time) and
observe how CrewAI handles tool errors. Then add a retry policy (look up
`max_retry_limit` on `Task`).

## E5 - Cost comparison

Add `usage_metrics=True` to the Crew (or enable it via config) and compute
how many tokens the CrewAI version uses versus the LangGraph and Agents SDK
versions for the same scenario. Log the numbers in `COMPARISON.md`.
