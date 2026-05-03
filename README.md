# Agentic AI - Practical Course

A hands-on, code-first course that teaches you to build production-shaped agents
by **building the same workflow-automation agent three times** in three different
frameworks, then comparing them head-to-head.

> Use case running through the whole course: a **"Personal Ops Assistant"** that
> triages your inbox, checks your calendar, talks to your CRM, and drafts
> replies - with a human-in-the-loop approval gate for anything risky.

---

## Why this structure works

Most agent tutorials teach one framework in isolation and leave you unable to
judge when it's the right tool. This course forces you to:

1. First, **feel the raw loop** by writing an agent from scratch in ~50 lines
   (Module 0) so you understand what frameworks abstract.
2. Then build the **same email/calendar/CRM agent** three ways - in LangGraph,
   OpenAI Agents SDK, and CrewAI - against a shared mock API.
3. Harden one version for production with evals, tracing, and prompt-injection
   defenses (Module 4).
4. Ship a capstone that talks to **real** Gmail + Google Calendar with a safety
   layer (Module 5).

By the end, you'll know which framework to reach for, and why.

---

## Course map

| Module | Topic                          | Time  | Runnable files                                       |
| ------ | ------------------------------ | ----- | ---------------------------------------------------- |
| 0      | Setup + raw ReAct loop         | 1 h   | `00-setup/hello_agent.py`                            |
| 1      | LangGraph                      | 4 h   | 3 lessons + `project_workflow_v1.py`                 |
| 2      | OpenAI Agents SDK              | 3 h   | 2 lessons + `project_workflow_v2.py`                 |
| 3      | CrewAI                         | 3 h   | 2 lessons + `project_workflow_v3.py`                 |
| 4      | Production: evals / tracing / security | 3 h | `evals.py`, `tracing_langfuse.py`, `security_prompt_injection.py` |
| 5      | Capstone: Personal Ops Assistant | 6-10 h | `05-capstone/` full app                           |

Finish with [COMPARISON.md](COMPARISON.md) for the side-by-side scorecard.

---

## Setup (5 minutes)

```bash
# 1. (Recommended) use uv - https://github.com/astral-sh/uv
#    Or use plain python -m venv + pip.
cd agentic-ai-course
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install the core + the modules you're working on. You can add more later.
pip install -e ".[langgraph]"           # when you start Module 1
pip install -e ".[langgraph,openai-agents]"   # Module 2
pip install -e ".[all]"                  # everything

# 3. Configure
cp .env.example .env
# then edit .env and paste your OPENAI_API_KEY

# 4. Smoke test
python 00-setup/hello_agent.py
```

If the hello-agent prints a tool-using conversation, you're wired up correctly.

---

## How to use each lesson

Each numbered folder has:

- `README.md` - concepts + walkthrough (~10 min read)
- `lesson_*.py` - runnable, heavily commented code (~30 min)
- `exercises.md` - 3-5 exercises; solutions in `solutions/`
- `project_workflow_v*.py` - module capstone using the shared mock APIs

Run any file standalone:

```bash
python 01-langgraph/lesson_1_graph_basics.py
```

All lessons use the utilities in [`shared/`](shared/):

- `shared/llm.py` - a thin OpenAI client you can swap for another provider
- `shared/mock_apis.py` - fake Email / Calendar / CRM SDKs with realistic
  latency, pagination and occasional errors
- `shared/fixtures/` - seed data (inbox, contacts, events) as JSON
- `shared/eval_harness.py` - LLM-as-judge + a tiny regression runner

---

## Mental model of an agent

```
                 +---------------------+
   user input -->|   LLM (controller)  |---> final answer
                 +----------+----------+
                            |
                      tool calls
                            v
     +-----------+   +-----------+   +-----------+
     |  search   |   |  email    |   |  calendar |   ... more tools
     +-----------+   +-----------+   +-----------+
                            |
                            v
                  observations back to LLM
                            |
                    (loop until done)
```

Every framework in this course implements this loop - they differ in **how they
let you structure the loop, share state, hand off between agents, and wire
humans into the middle**.

---

## Cost budget

With `gpt-4o-mini` (the default), running every lesson end-to-end from scratch
costs **under $0.50**. The capstone adds a few cents per run. Use a real model
(`gpt-4o` / `gpt-4.1`) only when you want to see quality differences.

---

## Progress tracker

Tick these off as you finish:

- [ ] Module 0: `hello_agent.py` runs and calls two tools
- [ ] Module 1: `project_workflow_v1.py` triages 3 emails end-to-end
- [ ] Module 2: `project_workflow_v2.py` does the same via handoffs
- [ ] Module 3: `project_workflow_v3.py` does it with a 3-agent crew
- [ ] Module 4: `evals.py` scores all three versions on the golden set
- [ ] Capstone: drafts a real Gmail reply (still in DRY_RUN mode!)
- [ ] Filled in your own opinions in `COMPARISON.md`

Good luck - and remember: **do the exercises**. Reading about agents is easy;
debugging one at 2am is how you actually learn.
