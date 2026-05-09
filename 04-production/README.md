# Module 4 - Production concerns

**Time:** ~3 hours. **Install:** `pip install -e ".[langgraph,production]"`.

Reading about agents is easy. Shipping one is hard. This module covers the
three things that turn a demo into a product:

1. **Evals** - automated regression testing of agent behaviour.
2. **Observability** - traces, cost, and latency you can actually search.
3. **Security** - prompt injection defenses, because your inbox agent will
   meet adversarial emails within week 1 of real use.

## Files

- **[`evals.py`](evals.py)** - run the golden-task suite in
  `shared/fixtures/golden_tasks.json` against your LangGraph v1 agent and
  print a pass/fail table.
- **[`tracing_langfuse.py`](tracing_langfuse.py)** - wire Langfuse into the
  v1 agent so every run appears on your dashboard with costs, tokens, and
  latencies.
- **[`security_prompt_injection.py`](security_prompt_injection.py)** -
  demonstrate an attack, then three layers of defense.

## Why only the LangGraph version?

Module 4 focuses on one version so the lessons stay short. The techniques
port to Agents SDK and CrewAI 1:1 - you just change the runner function
passed into the harness.

## Checklist

- [ ] You ran `evals.py` and got a score on the golden suite.
- [ ] You can point at a Langfuse trace and explain what each span means.
- [ ] You can describe three distinct defenses against email-borne prompt
      injection (content quarantine, tool allowlisting, output filtering).
