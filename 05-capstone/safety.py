"""Central safety layer for the capstone.

Every write action (send email, create event) goes through functions defined
here. Two-factor gate:

  1. Environment: `CAPSTONE_DRY_RUN=true` (default) short-circuits to a dry
     return. You have to flip it deliberately.
  2. Per-action: a human CLI approval is required even when DRY_RUN is false.

The LLM never calls these directly; it calls tools that delegate here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def dry_run_enabled() -> bool:
    return os.getenv("CAPSTONE_DRY_RUN", "true").strip().lower() in {"1", "true", "yes"}


@dataclass
class ApprovalRequest:
    action: str
    summary: str
    payload: dict[str, Any]
    requested_at: str

    def render(self) -> str:
        payload_str = "\n".join(f"  {k}: {v}" for k, v in self.payload.items())
        return (
            f"\n==== APPROVAL REQUIRED ====\n"
            f"When   : {self.requested_at}\n"
            f"Action : {self.action}\n"
            f"Summary: {self.summary}\n"
            f"Payload:\n{payload_str}\n"
            f"==========================="
        )


def request_approval(req: ApprovalRequest) -> bool:
    """Block on the terminal until the user types 'y' or 'n'.

    Callers can override the input source for testing by setting the env var
    CAPSTONE_AUTO_APPROVE to 'y' or 'n'.
    """
    auto = os.getenv("CAPSTONE_AUTO_APPROVE")
    if auto is not None:
        return auto.strip().lower() in {"y", "yes"}

    print(req.render())
    choice = input("Approve? [y/N] ").strip().lower() or "n"
    return choice in {"y", "yes"}


def guard_send_email(to: str, subject: str, body: str, sender_allowlist: list[str]) -> dict[str, Any]:
    """Return the safety verdict for a proposed send.

    Rules:
      - `to` must be in `sender_allowlist` (derived from the ORIGINAL From of
        a message already in your inbox). This prevents the LLM from inventing
        recipients.
      - DRY_RUN short-circuits with `will_send=False`.
      - Otherwise we require human approval.
    """
    if to.strip().lower() not in {a.strip().lower() for a in sender_allowlist}:
        return {"will_send": False, "reason": f"recipient {to!r} not in allowlist"}

    if dry_run_enabled():
        return {"will_send": False, "reason": "dry-run", "preview": {"to": to, "subject": subject, "body": body}}

    approved = request_approval(
        ApprovalRequest(
            action="send_email",
            summary=f"send to {to}",
            payload={"to": to, "subject": subject, "body_preview": body[:240]},
            requested_at=datetime.utcnow().isoformat() + "Z",
        )
    )
    if not approved:
        return {"will_send": False, "reason": "denied_by_human"}
    return {"will_send": True}


def guard_create_event(
    title: str,
    start_iso: str,
    end_iso: str,
    attendees: list[str],
) -> dict[str, Any]:
    if dry_run_enabled():
        return {
            "will_create": False,
            "reason": "dry-run",
            "preview": {"title": title, "start": start_iso, "end": end_iso, "attendees": attendees},
        }
    approved = request_approval(
        ApprovalRequest(
            action="create_event",
            summary=f"{title} @ {start_iso}",
            payload={"start": start_iso, "end": end_iso, "attendees": attendees},
            requested_at=datetime.utcnow().isoformat() + "Z",
        )
    )
    if not approved:
        return {"will_create": False, "reason": "denied_by_human"}
    return {"will_create": True}
