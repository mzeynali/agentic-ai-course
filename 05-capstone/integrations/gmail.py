"""Minimal Gmail integration using google-api-python-client.

Scopes chosen to be the MINIMUM needed:
  - gmail.readonly       : list/get messages
  - gmail.compose        : create drafts (note: NOT gmail.send)

We deliberately don't request gmail.send; the app can create drafts, and you
promote a draft to a send via safety.py's approval flow + a separate call.
"""

from __future__ import annotations

import base64
import os
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
TOKEN_PATH = Path.home() / ".ops_assistant_gmail_token.json"


def _load_credentials() -> Credentials:
    client_secrets = os.getenv("GOOGLE_OAUTH_CLIENT_SECRETS")
    if not client_secrets or not Path(client_secrets).exists():
        raise RuntimeError(
            "Set GOOGLE_OAUTH_CLIENT_SECRETS to the path of your OAuth client_secret.json. "
            "See 05-capstone/README.md for setup."
        )

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
    return build("gmail", "v1", credentials=_load_credentials(), cache_discovery=False)


def list_unread(limit: int = 10) -> list[dict[str, Any]]:
    svc = _service()
    resp = (
        svc.users()
        .messages()
        .list(userId="me", q="is:unread", maxResults=limit)
        .execute()
    )
    ids = [m["id"] for m in resp.get("messages", [])]
    out: list[dict[str, Any]] = []
    for mid in ids:
        out.append(get_message(mid))
    return out


def get_message(msg_id: str) -> dict[str, Any]:
    svc = _service()
    msg = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}

    def _walk_body(payload: dict[str, Any]) -> str:
        """Return the first text/plain body we find, else empty."""
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        for part in payload.get("parts", []) or []:
            found = _walk_body(part)
            if found:
                return found
        return ""

    body = _walk_body(msg["payload"])
    return {
        "id": msg_id,
        "thread_id": msg.get("threadId"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "body": body[:4000],  # truncate huge messages
        "snippet": msg.get("snippet", ""),
    }


def create_draft(to: str, subject: str, body: str, thread_id: str | None = None) -> str:
    """Create a Gmail draft and return its id. Nothing is sent."""
    svc = _service()
    mime = MIMEText(body, "plain", "utf-8")
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    draft_body: dict[str, Any] = {"message": {"raw": raw}}
    if thread_id:
        draft_body["message"]["threadId"] = thread_id
    draft = svc.users().drafts().create(userId="me", body=draft_body).execute()
    return draft["id"]


def send_draft(draft_id: str) -> dict[str, Any]:
    """Promote a draft to a send. ONLY called from safety-guarded code."""
    svc = _service()
    return svc.users().drafts().send(userId="me", body={"id": draft_id}).execute()
