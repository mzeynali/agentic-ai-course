# Module 0 - Setup and the Mental Model of an Agent

**Time:** ~1 hour. **Goal:** understand the ReAct loop that every framework in
this course abstracts, so you can debug them when they misbehave.

## 1. Install and configure

From the course root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .           # just the core dependencies
cp .env.example .env       # then paste your OPENAI_API_KEY
```

Test your setup:

```bash
python 00-setup/hello_agent.py
```

If you see a multi-turn conversation where the model calls `get_weather` and
`get_time` tools and produces a final answer, you're good.

## 2. What is an "agent"?

Strip away the buzzwords and an LLM agent is just a loop:

```text
repeat:
    messages.append(user_input)
    response = LLM(messages, tools=available_tools)
    if response has tool_calls:
        for each call:
            result = call_tool(call.name, call.args)
            messages.append(tool_result)
    else:
        return response.text   # final answer
```

That's it. Everything else - LangGraph's state machine, CrewAI's multi-agent
orchestration, OpenAI Agents SDK's handoffs - is sugar on top of this loop.

## 3. Key concepts you'll meet in every module

| Concept            | One-sentence definition                                                 | First appears |
| ------------------ | ----------------------------------------------------------------------- | ------------- |
| **Tool calling**   | LLM emits a structured function call; your code runs it and returns the result. | Module 0      |
| **ReAct**          | Reason/Act loop: think -> call a tool -> observe -> think again.        | Module 0      |
| **Short-term memory** | The current message list inside one conversation.                      | Module 0      |
| **Long-term memory**  | A separate store (DB / vector store) that survives restarts and conversations. | Module 1 L3 |
| **Planning**       | Emitting a multi-step plan before executing any tool.                   | Module 1      |
| **Reflection**     | Critiquing your own output and re-trying.                               | Module 3      |
| **Handoff**        | One agent delegating the rest of the task to a different agent.         | Module 2      |
| **HITL**           | Human-in-the-loop: pause for approval before a risky action.            | Module 1 L2   |
| **Guardrails**     | Pre/post-checks on input or output (e.g. "don't output PII").           | Module 2      |
| **Tracing**        | Recording every LLM call + tool call for debugging and cost attribution. | Module 4     |

## 4. Why we start from scratch

Frameworks hide the loop. That's great when everything works and awful when it
doesn't. After reading `hello_agent.py` (it's ~50 lines of real logic) you will
always be able to answer: "what is the LLM actually seeing right now?" - which
is the single most useful debugging skill for agents.

## 5. Checklist before moving on

- [ ] `python 00-setup/hello_agent.py` runs end-to-end.
- [ ] You can explain, without looking, what the three possible outcomes of one
      iteration of the loop are (final answer / single tool call / multiple
      parallel tool calls).
- [ ] You understand why the **entire** message history, including past tool
      results, has to be sent back to the model on every iteration.

When those three boxes are ticked, move on to `01-langgraph/`.
