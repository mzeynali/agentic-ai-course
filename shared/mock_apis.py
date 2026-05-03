"""In-memory Email / Calendar / CRM 'SDKs' used by every lesson.

Design goals:
  - Realistic enough to feel like a real API (ids, timestamps, pagination,
    occasional transient errors) without needing network access or credentials.
  - Deterministic by default (seeded RNG) so evals are reproducible; pass
    flaky=True to get realistic error injection.
  - Pure Python, zero dependencies.

Three services:
  EmailAPI     - list_inbox, get_email, draft_reply, send_draft
  CalendarAPI  - list_events, find_free_slots, create_event
  CRMAPI       - search_contacts, get_contact, log_activity
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TransientAPIError(RuntimeError):
    """Raised by the mock APIs to simulate transient failures.

    Good agents should retry these; brittle ones will crash. Use this in
    Module 4 to test resilience.
    """


def _load_fixture(name: str) -> list[dict[str, Any]]:
    path = FIXTURES_DIR / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class _FlakyMixin:
    """Injects latency and the occasional transient error."""

    flaky: bool = False
    _seed: int = 42
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self._seed)

    def _tick(self, fail_rate: float = 0.0) -> None:
        # Gentle latency so lessons feel real but don't bore the learner.
        time.sleep(self._rng.uniform(0.01, 0.05))
        if self.flaky and self._rng.random() < fail_rate:
            raise TransientAPIError("mock API: transient failure, please retry")


# -----------------------------------------------------------------------------
# Email
# -----------------------------------------------------------------------------


@dataclass
class EmailAPI(_FlakyMixin):
    """Fake Gmail-ish API seeded from shared/fixtures/inbox.json."""

    _inbox: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _drafts: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _sent: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._inbox = _load_fixture("inbox.json")

    def list_inbox(self, unread_only: bool = True, limit: int = 10) -> list[dict[str, Any]]:
        self._tick(fail_rate=0.05)
        items = [e for e in self._inbox if (not unread_only or e.get("unread", True))]
        return items[:limit]

    def get_email(self, email_id: str) -> dict[str, Any]:
        self._tick(fail_rate=0.02)
        for e in self._inbox:
            if e["id"] == email_id:
                return e
        raise ValueError(f"email {email_id!r} not found")

    def mark_read(self, email_id: str) -> None:
        self._tick()
        for e in self._inbox:
            if e["id"] == email_id:
                e["unread"] = False
                return
        raise ValueError(f"email {email_id!r} not found")

    def draft_reply(self, email_id: str, body: str, subject: str | None = None) -> str:
        """Create a draft reply. Returns draft_id. Does NOT send."""
        self._tick(fail_rate=0.02)
        original = self.get_email(email_id)
        draft_id = f"draft_{uuid.uuid4().hex[:8]}"
        self._drafts[draft_id] = {
            "id": draft_id,
            "in_reply_to": email_id,
            "to": original["from"],
            "subject": subject or f"Re: {original['subject']}",
            "body": body,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        return draft_id

    def send_draft(self, draft_id: str) -> dict[str, Any]:
        """Actually 'send' a draft. This is the high-stakes action that should
        always sit behind a human approval gate in real systems.
        """
        self._tick(fail_rate=0.05)
        if draft_id not in self._drafts:
            raise ValueError(f"draft {draft_id!r} not found")
        msg = self._drafts.pop(draft_id)
        msg["sent_at"] = datetime.utcnow().isoformat() + "Z"
        self._sent.append(msg)
        return msg

    # Introspection helpers used by lessons and evals.
    def sent_messages(self) -> list[dict[str, Any]]:
        return list(self._sent)

    def drafts(self) -> list[dict[str, Any]]:
        return list(self._drafts.values())


# -----------------------------------------------------------------------------
# Calendar
# -----------------------------------------------------------------------------


@dataclass
class CalendarAPI(_FlakyMixin):
    _events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._events = _load_fixture("events.json")

    def list_events(self, start: str, end: str) -> list[dict[str, Any]]:
        """Return events that overlap [start, end) (ISO 8601 strings)."""
        self._tick(fail_rate=0.02)
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        out = []
        for ev in self._events:
            es = datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))
            ee = datetime.fromisoformat(ev["end"].replace("Z", "+00:00"))
            if es < e and ee > s:
                out.append(ev)
        return sorted(out, key=lambda x: x["start"])

    def find_free_slots(
        self,
        start: str,
        end: str,
        duration_minutes: int = 30,
        working_hours: tuple[int, int] = (9, 17),
    ) -> list[dict[str, str]]:
        """Propose free slots in [start, end) during working hours."""
        self._tick(fail_rate=0.02)
        busy = self.list_events(start, end)
        busy_ranges = [
            (
                datetime.fromisoformat(ev["start"].replace("Z", "+00:00")),
                datetime.fromisoformat(ev["end"].replace("Z", "+00:00")),
            )
            for ev in busy
        ]
        cursor = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        step = timedelta(minutes=duration_minutes)
        work_start, work_end = working_hours
        slots: list[dict[str, str]] = []
        while cursor + step <= end_dt and len(slots) < 8:
            in_hours = work_start <= cursor.hour < work_end and cursor.weekday() < 5
            clashes = any(not (cursor + step <= bs or cursor >= be) for bs, be in busy_ranges)
            if in_hours and not clashes:
                slots.append(
                    {
                        "start": cursor.isoformat() + "Z",
                        "end": (cursor + step).isoformat() + "Z",
                    }
                )
                cursor += step
            else:
                cursor += timedelta(minutes=15)
        return slots

    def create_event(
        self,
        title: str,
        start: str,
        end: str,
        attendees: list[str] | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        self._tick(fail_rate=0.05)
        event = {
            "id": f"evt_{uuid.uuid4().hex[:8]}",
            "title": title,
            "start": start,
            "end": end,
            "attendees": attendees or [],
            "description": description,
        }
        self._events.append(event)
        return event


# -----------------------------------------------------------------------------
# CRM
# -----------------------------------------------------------------------------


@dataclass
class CRMAPI(_FlakyMixin):
    _contacts: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _activities: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._contacts = _load_fixture("contacts.json")

    def search_contacts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        self._tick(fail_rate=0.02)
        q = query.lower().strip()
        hits = [
            c
            for c in self._contacts
            if q in c["name"].lower()
            or q in c["email"].lower()
            or q in c.get("company", "").lower()
        ]
        return hits[:limit]

    def get_contact(self, contact_id: str) -> dict[str, Any]:
        self._tick()
        for c in self._contacts:
            if c["id"] == contact_id:
                return c
        raise ValueError(f"contact {contact_id!r} not found")

    def log_activity(self, contact_id: str, kind: str, note: str) -> dict[str, Any]:
        self._tick(fail_rate=0.02)
        _ = self.get_contact(contact_id)  # validate
        activity = {
            "id": f"act_{uuid.uuid4().hex[:8]}",
            "contact_id": contact_id,
            "kind": kind,
            "note": note,
            "at": datetime.utcnow().isoformat() + "Z",
        }
        self._activities.append(activity)
        return activity

    def activities(self, contact_id: str | None = None) -> list[dict[str, Any]]:
        if contact_id is None:
            return list(self._activities)
        return [a for a in self._activities if a["contact_id"] == contact_id]


# -----------------------------------------------------------------------------
# Convenience factory
# -----------------------------------------------------------------------------


@dataclass
class Workspace:
    """Bundle the three APIs with a single seed so lessons can do:

        ws = Workspace()
        ws.email.list_inbox()
    """

    flaky: bool = False
    seed: int = 42
    email: EmailAPI = field(init=False)
    calendar: CalendarAPI = field(init=False)
    crm: CRMAPI = field(init=False)

    def __post_init__(self) -> None:
        self.email = EmailAPI(flaky=self.flaky, _seed=self.seed)
        self.calendar = CalendarAPI(flaky=self.flaky, _seed=self.seed + 1)
        self.crm = CRMAPI(flaky=self.flaky, _seed=self.seed + 2)
