# Framework comparison: LangGraph vs OpenAI Agents SDK vs CrewAI

A scorecard to fill in **as you go** - not after the fact. After finishing
each module you should update the relevant column with your own notes.
Opinions in this file are starting points, not gospel.

## TL;DR (my opinion - form your own)

| Dimension                      | LangGraph       | OpenAI Agents SDK | CrewAI           |
| ------------------------------ | --------------- | ----------------- | ---------------- |
| Lines to get "hello agent"     | Medium          | **Low**           | Medium           |
| Explicit control flow          | **Yes (graph)** | No (implicit)     | No (process enum) |
| Multi-agent ergonomics         | Manual wiring   | **Handoffs feel clean** | **Roles-and-tasks is intuitive** |
| Human-in-the-loop primitive    | **`interrupt()` built-in** | Roll your own     | Roll your own    |
| Memory (short + long term)     | **Checkpointer + Store** | Basic per-run context | `memory=True` on crew |
| Streaming / progressive UI     | Yes (`stream`)  | **Yes (rich events)** | Yes (verbose logs) |
| Tracing / observability        | LangSmith / Langfuse | **Native in platform.openai.com** | Langfuse / stdout |
| Lock-in to OpenAI              | Provider-agnostic | **Strong** (name in title) | Provider-agnostic |
| Debuggability when broken      | **Excellent - you can inspect state per node** | OK | Weak - agents talk in prose |
| Ceiling for custom control flow| **Very high**   | Medium            | Medium           |
| Time to ship a simple demo     | Medium          | **Short**         | Short            |

### When to pick which

- **LangGraph** when you need HITL, durable state, or complex branching. The
  capstone uses LangGraph for exactly this reason.
- **OpenAI Agents SDK** when you want the fastest path to a working
  multi-agent handoff pattern and you're already all-in on OpenAI. Great for
  internal prototypes and tools.
- **CrewAI** when the work naturally decomposes into **roles** (researcher,
  writer, critic) and you want the team-meeting framing to shape thinking.
  Also good for non-technical stakeholders reading the code.

## Fill-in-as-you-go rubric (1-5, higher = better)

| Dimension                    | LangGraph | Agents SDK | CrewAI | Notes |
| ---------------------------- | --------- | ---------- | ------ | ----- |
| Developer experience         |           |            |        |       |
| Debuggability                |           |            |        |       |
| Streaming ergonomics         |           |            |        |       |
| HITL support                 |           |            |        |       |
| Multi-agent handoff clarity  |           |            |        |       |
| Short-term memory            |           |            |        |       |
| Long-term memory             |           |            |        |       |
| Production-readiness (evals / tracing / retries) |  |  |  |    |
| Lock-in risk                 |           |            |        |       |
| Token overhead (see E5 M3)   |           |            |        |       |
| **Total**                    |           |            |        |       |

## Scenarios I tested this on

Fill in once you have run the same inputs through all three versions of the
Personal Ops Assistant:

- [ ] 5 unread inbox, one phishing injection, one scheduling request
- [ ] Empty inbox - does the agent gracefully say "nothing to do"?
- [ ] Flaky API (`Workspace(flaky=True)`) - how does each framework handle
      transient errors? Which retries and which crashes?

## Token / latency / cost comparison

Record results from enabling usage tracking in each framework:

| Framework     | Input tokens | Output tokens | Wall time | $ (gpt-4o-mini) |
| ------------- | ------------ | ------------- | --------- | ---------------- |
| LangGraph     |              |               |           |                  |
| Agents SDK    |              |               |           |                  |
| CrewAI        |              |               |           |                  |

## One-sentence summary to write in your own words

> Write your verdict here after finishing all modules. If you struggle to
> write a single sentence, go back and finish the exercises - the opinion
> will form itself.
