"""Run the golden-task suite against the LangGraph v1 agent.

Install:
    pip install -e ".[langgraph,production]"

Run:
    python 04-production/evals.py

Swap `runner_for_langgraph` for an Agents-SDK or CrewAI runner to evaluate
those versions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Reuse the graph, tools, and SYSTEM_PROMPT from the Module 1 project.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "01-langgraph")
)
from project_workflow_v1 import SYSTEM_PROMPT, build_graph  # noqa: E402

from shared.eval_harness import GoldenTask, load_golden_tasks, run_suite  # noqa: E402


def runner_for_langgraph(task: GoldenTask) -> list[dict]:
    """Execute the LangGraph v1 agent on a single golden task and return a
    cleaned transcript the judge can read.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": f"eval-{task.id}"}}
    initial = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=task.input),
        ]
    }
    # Use invoke (not stream) and disable HITL approvals by auto-denying -
    # this means the agent can never actually SEND; it can only draft.
    # That's the right policy for regression evals.
    from langgraph.types import Command  # local import for clarity

    try:
        graph.invoke(initial, config=config)
    except Exception as exc:  # noqa: BLE001
        return [{"role": "error", "content": repr(exc)}]

    # If the run paused on an interrupt (send_draft approval), auto-deny and
    # resume. For golden tasks whose rubric forbids send, this is the correct
    # behavior.
    state = graph.get_state(config)
    while state.interrupts:
        graph.invoke(Command(resume="deny"), config=config)
        state = graph.get_state(config)

    final = graph.get_state(config)
    cleaned: list[dict] = []
    for m in final.values.get("messages", []):
        if isinstance(m, AIMessage):
            cleaned.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {"name": c["name"], "args": c["args"]} for c in (m.tool_calls or [])
                    ],
                }
            )
        elif isinstance(m, ToolMessage):
            cleaned.append(
                {
                    "role": "tool",
                    "name": m.name,
                    "content": (m.content or "")[:400],
                }
            )
        elif isinstance(m, HumanMessage):
            cleaned.append({"role": "user", "content": m.content})
        elif isinstance(m, SystemMessage):
            continue  # no need to send the system prompt to the judge
    return cleaned


def main() -> None:
    tasks = load_golden_tasks()
    print(f"Running {len(tasks)} golden tasks against the LangGraph v1 agent...\n")
    results = run_suite(runner_for_langgraph, tasks=tasks)

    print("\n==== SCORECARD ====")
    print(f"{'task':<10} {'kind':<12} {'score':<6} {'passed':<7} reason")
    print("-" * 80)
    for r in results:
        task = next(t for t in tasks if t.id == r.task_id)
        print(
            f"{r.task_id:<10} {task.kind:<12} {r.score:<6} "
            f"{'YES' if r.passed else 'NO':<7} {r.reason[:60]}"
        )


if __name__ == "__main__":
    main()
