# Module 1 - Exercises

Do these in order. Solutions are in `solutions/`; don't peek until you've
written your own.

## E1 - Add a `summarize_day` tool to lesson 1

Extend `lesson_1_graph_basics.py` with a third tool that returns a short
"daily brief" string given a date. Re-run the graph with a prompt like
`"Give me a daily brief and the weather in Tehran."` and verify the model
calls the new tool.

## E2 - Add a recursion limit

LangGraph has a built-in `recursion_limit` kwarg on `graph.invoke()`. Set it
to `3` and craft a pathological prompt that makes the agent loop forever
(hint: ask it to keep calling `get_time` until a condition that is never
true). Observe the error. Why is this safer than the naive `while True`
loop from Module 0?

## E3 - Route by intent

In `lesson_2_tools_hitl.py`, add a *router* node before the LLM node that
classifies the user's goal into one of `{"triage", "schedule", "reply"}`
and routes to a different system prompt for each. You should end up with a
graph like:

```text
START -> router -> {triage_llm | schedule_llm | reply_llm} -> tools -> ... -> END
```

## E4 - Resume from crash

Run `lesson_2_tools_hitl.py`, hit the approval prompt, then Ctrl-C. Now
write a short script that:

1. Rebuilds the graph with the **same** `MemorySaver` instance.
2. Calls `graph.get_state(config)` with the same `thread_id`.
3. Inspects the interrupts and resumes the run without starting over.

(In production you'd use `SqliteSaver` or `PostgresSaver` for real
durability.)

## E5 - Structured output for the final answer

Modify `project_workflow_v1.py` so the agent's **final** message is a JSON
object with fields `{emails_replied: list[str], emails_skipped: list[str],
events_created: list[str]}`. Hint: use `ChatOpenAI(...).with_structured_output`
on a *second* LLM that runs only at the end via an explicit graph node.

## E6 - A new golden task

Add a new entry to `shared/fixtures/golden_tasks.json` that tests a
scenario you care about. Run it through the eval harness in Module 4 later
to see if your LangGraph agent handles it.
