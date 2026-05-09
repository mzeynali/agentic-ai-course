"""Minimal Google Calendar integration.

Scopes:
  - calendar.readonly : list events, compute free/busy
  - calendar.events   : create events (guarded via safety.py)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]
TOKEN_PATH = Path.home() / ".ops_assistant_gcal_token.json"


def _load_credentials() -> Credentials:
    client_secrets = os.getenv("GOOGLE_OAUTH_CLIENT_SECRETS")
    if not client_secrets or not Path(client_secrets).exists():
        raise RuntimeError("Set GOOGLE_OAUTH_CLIENT_SECRETS - see 05-capstone/README.md.")
    creds: Credentials | None = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def _service():
    return build("calendar", "v3", credentials=_load_credentials(), cache_discovery=False)


def list_events(start_iso: str, end_iso: str) -> list[dict[str, Any]]:
    svc = _service()
    resp = (
        svc.events()
        .list(
            calendarId="primary",
            timeMin=start_iso,
            timeMax=end_iso,
            singleEvents=True,
            orderBy="startTime",
            maxResults=50,
        )
        .execute()
    )
    return resp.get("items", [])


def find_free_slots(
    start_iso: str,
    end_iso: str,
    duration_minutes: int = 30,
    working_hours: tuple[int, int] = (9, 17),
) -> list[dict[str, str]]:
    events = list_events(start_iso, end_iso)
    busy = []
    for ev in events:
        s = ev["start"].get("dateTime") or ev["start"].get("date")
        e = ev["end"].get("dateTime") or ev["end"].get("date")
        if s and e:
            busy.append((_parse(s), _parse(e)))

    cursor = _parse(start_iso)
    end_dt = _parse(end_iso)
    step = timedelta(minutes=duration_minutes)
    wh_start, wh_end = working_hours
    slots: list[dict[str, str]] = []
    while cursor + step <= end_dt and len(slots) < 8:
        in_hours = wh_start <= cursor.hour < wh_end and cursor.weekday() < 5
        clashes = any(not (cursor + step <= bs or cursor >= be) for bs, be in busy)
        if in_hours and not clashes:
            slots.append(
                {
                    "start": cursor.isoformat(),
                    "end": (cursor + step).isoformat(),
                }
            )
            cursor += step
        else:
            cursor += timedelta(minutes=15)
    return slots


def create_event(
    title: str,
    start_iso: str,
    end_iso: str,
    attendees: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Create a real calendar event. ONLY call from safety-guarded code."""
    svc = _service()
    body = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
        "attendees": [{"email": a} for a in (attendees or [])],
    }
    return svc.events().insert(calendarId="primary", body=body, sendUpdates="all").execute()


def _parse(s: str) -> datetime:
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
