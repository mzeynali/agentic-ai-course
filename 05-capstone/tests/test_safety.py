"""Unit tests for safety.py - the most important file in the capstone.

Run with:
    pytest 05-capstone/tests/test_safety.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from safety import dry_run_enabled, guard_create_event, guard_send_email


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("CAPSTONE_DRY_RUN", raising=False)
    monkeypatch.delenv("CAPSTONE_AUTO_APPROVE", raising=False)


def test_dry_run_default_is_true(monkeypatch):
    monkeypatch.delenv("CAPSTONE_DRY_RUN", raising=False)
    assert dry_run_enabled() is True


def test_dry_run_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CAPSTONE_DRY_RUN", "false")
    assert dry_run_enabled() is False


def test_send_blocked_when_recipient_not_in_allowlist(monkeypatch):
    monkeypatch.setenv("CAPSTONE_DRY_RUN", "false")
    monkeypatch.setenv("CAPSTONE_AUTO_APPROVE", "y")
    verdict = guard_send_email(
        to="attacker@evil.com",
        subject="bait",
        body="hi",
        sender_allowlist=["sara@acme.com"],
    )
    assert verdict["will_send"] is False
    assert "allowlist" in verdict["reason"].lower()


def test_send_dry_run_never_sends(monkeypatch):
    monkeypatch.setenv("CAPSTONE_DRY_RUN", "true")
    verdict = guard_send_email(
        to="sara@acme.com",
        subject="ok",
        body="hi",
        sender_allowlist=["sara@acme.com"],
    )
    assert verdict["will_send"] is False
    assert verdict["reason"] == "dry-run"


def test_send_with_approval(monkeypatch):
    monkeypatch.setenv("CAPSTONE_DRY_RUN", "false")
    monkeypatch.setenv("CAPSTONE_AUTO_APPROVE", "y")
    verdict = guard_send_email(
        to="sara@acme.com",
        subject="ok",
        body="hi",
        sender_allowlist=["sara@acme.com"],
    )
    assert verdict["will_send"] is True


def test_send_denied_by_human(monkeypatch):
    monkeypatch.setenv("CAPSTONE_DRY_RUN", "false")
    monkeypatch.setenv("CAPSTONE_AUTO_APPROVE", "n")
    verdict = guard_send_email(
        to="sara@acme.com",
        subject="ok",
        body="hi",
        sender_allowlist=["sara@acme.com"],
    )
    assert verdict["will_send"] is False
    assert verdict["reason"] == "denied_by_human"


def test_create_event_dry_run(monkeypatch):
    monkeypatch.setenv("CAPSTONE_DRY_RUN", "true")
    verdict = guard_create_event(
        title="Intro call",
        start_iso="2026-04-30T14:00:00Z",
        end_iso="2026-04-30T14:30:00Z",
        attendees=["sara@acme.com"],
    )
    assert verdict["will_create"] is False
    assert verdict["reason"] == "dry-run"


def test_create_event_approved(monkeypatch):
    monkeypatch.setenv("CAPSTONE_DRY_RUN", "false")
    monkeypatch.setenv("CAPSTONE_AUTO_APPROVE", "y")
    verdict = guard_create_event(
        title="Intro call",
        start_iso="2026-04-30T14:00:00Z",
        end_iso="2026-04-30T14:30:00Z",
        attendees=["sara@acme.com"],
    )
    assert verdict["will_create"] is True
