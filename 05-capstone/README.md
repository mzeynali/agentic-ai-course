# Module 5 - Capstone: Personal Ops Assistant

**Time:** 6-10 hours. **Install:**

```bash
pip install -e ".[langgraph,production,capstone]"
```

You've now built the same agent three times. Time to pick your favorite
framework and ship something that talks to real services.

> **This capstone ships with LangGraph as the default framework** because its
> `interrupt()` maps cleanly to the HITL approval gate. If you preferred
> Agents SDK or CrewAI, port
> [`capstone_agent.py`](capstone_agent.py) - the tool interfaces and safety
> layer are framework-agnostic.

## Brief

Build a CLI app `ops` that:

1. Reads your Gmail (via OAuth) and Google Calendar.
2. Talks to a local "CRM" (SQLite-backed `integrations/crm_store.py`) or
   optionally HubSpot Free.
3. Can: `triage`, `propose-replies`, `send` (only with explicit
   `--approve`), `schedule` meetings.
4. Always runs in `DRY_RUN=true` mode by default: no mail is sent and no
   calendar events are created unless the user flips the env var AND types
   `y` at the approval prompt.
5. Emits Langfuse traces so you can debug in production.

You will have successfully completed the capstone when:

- `ops triage` summarizes your real inbox without sending anything.
- `ops propose-replies` generates drafts that appear as **drafts** in Gmail
  (not sent).
- `ops schedule "30min coffee with <someone> next week"` proposes real free
  slots and, with `--approve`, creates a calendar event.
- `pytest 05-capstone/tests/` passes.
- Your `evals.py` on the golden tasks hits >= 16/20 total score.

## Deliverable layout

```text
05-capstone/
├── README.md            # this file
├── capstone_agent.py    # the LangGraph agent + tools
├── cli.py               # Typer CLI entry point
├── safety.py            # HITL approval, DRY_RUN gating, send-allowlist
├── integrations/
│   ├── __init__.py
│   ├── gmail.py         # OAuth + list/get/draft/send (drafts by default)
│   ├── gcal.py          # OAuth + free/busy + event create
│   └── crm_store.py     # SQLite-backed mini CRM (HubSpot-shaped schema)
└── tests/
    ├── test_safety.py   # unit tests for the approval layer
    └── test_crm.py      # CRUD round-trip on the CRM
```

Files are created for you as **scaffolds** - they're runnable but intentionally
incomplete in places marked with `# TODO(you)`. Finish them as you go.

## Setup in 5 steps

1. **Create a Google Cloud project** and enable the Gmail + Calendar APIs.
2. Download `client_secret.json` for an **OAuth 2.0 Desktop app** credential
   and put it at a path you'll reference via
   `GOOGLE_OAUTH_CLIENT_SECRETS=/path/to/client_secret.json`.
3. (Optional) Sign up for [Langfuse](https://cloud.langfuse.com) and paste
   your keys into `.env` (see `.env.example` at the course root).
4. Run `python 05-capstone/cli.py init` - this does the OAuth flow in your
   browser and caches tokens at `~/.ops_assistant_token.json`.
5. Run `python 05-capstone/cli.py triage` to see your unread mail
   summarized.

## Safety rubric (self-grade while building)

You pass the capstone only if every one of these is true:

- [ ] Sending email requires both `CAPSTONE_DRY_RUN=false` AND a typed
      approval at the CLI. (`safety.py` enforces this centrally.)
- [ ] The tool surface exposed to the LLM does NOT include a direct
      "send email to any address" tool. The LLM can only target the
      original sender of an email already in your inbox.
- [ ] Email content is quarantined (see Module 4 defense D3) before being
      handed to the LLM.
- [ ] `pytest 05-capstone/tests/` passes.
- [ ] Langfuse shows at least one trace with full tool call detail.

## Rubric for self-grading

| Aspect                                | Points |
| ------------------------------------- | -----: |
| Agent works end-to-end on real inbox  |      5 |
| HITL gate prevents accidental sends   |      5 |
| Golden-task eval >= 16/20             |      3 |
| Tests pass                            |      3 |
| Langfuse tracing wired in             |      2 |
| Clear README for a new user           |      2 |
| **Total**                             |  **20** |

A score of 16+ means you could hand this off to a colleague and they could
use it without reading your mind.

## Where to go from here

- Port the capstone to Agents SDK and compare token counts (see `COMPARISON.md`).
- Add a second data source (Slack, Notion, a vector store over your docs).
- Swap Gmail for your work email (M365 Graph API uses the same pattern).
- Turn the CLI into a small FastAPI service and have it run on a cron.
